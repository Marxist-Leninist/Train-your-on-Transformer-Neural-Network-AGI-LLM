#!/usr/bin/env python3
# 5o_plus.py — joint AR+NAT+SAT trainer/decoder with strict Chinchilla budgeting,
# robust resume/warm-start, LR scheduling, OOM backoff, and exact step math.

from __future__ import annotations
import argparse, json, math, os, pathlib, random, time
from contextlib import nullcontext
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, logging as hf_log
from tqdm.auto import tqdm

# ───────────────────────── Globals ─────────────────────────
hf_log.set_verbosity_error()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "Qwen/Qwen3-235B-A22B-Thinking-2507")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
VOCAB = max(tok.get_vocab().values()) + 1
BLANK = tok.pad_token_id
EOS = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

PRESETS: Dict[str, Dict[str, int]] = {
    "small": dict(d=512, layers=8, heads=16, rank=64),
    "base":  dict(d=768, layers=12, heads=24, rank=96),
}

DEFAULT_BLOCK = 576
SAT_BLOCK = 2
EMIT_LAMBDA = 0.1
CKDIR = pathlib.Path("ckpts_joint")

# ───────────────────────── Utils ─────────────────────────
def rng_state():
    if DEV.type == "cuda":
        try:
            return torch.cuda.get_rng_state(DEV)
        except TypeError:
            return torch.cuda.get_rng_state()
    return torch.get_rng_state()

def _is_probably_ckpt(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and path.suffix == ".pt" and not path.name.endswith(".pt.tmp") and path.stat().st_size > (1<<20)
    except Exception:
        return False

def _resolve_ckpt(path: pathlib.Path) -> pathlib.Path | None:
    try:
        if path.is_dir():
            cands = sorted([p for p in path.glob("*.pt") if _is_probably_ckpt(p)],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None
        if path.suffix == ".tmp":
            solid = path.with_suffix("")
            return solid if _is_probably_ckpt(solid) else _resolve_ckpt(path.parent)
        return path if _is_probably_ckpt(path) else _resolve_ckpt(path.parent)
    except Exception:
        return None

def _try_load(path: pathlib.Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"[ckpt-skip] {path} not usable: {e}")
        return None

# AMP helper
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _auto_amp_dtype():
    if DEV.type == "cuda":
        try:
            maj, _ = torch.cuda.get_device_capability()
            return torch.float16 if maj < 8 else torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float32

def amp(enabled):
    return nullcontext() if not enabled else _ac(device_type="cuda", dtype=_auto_amp_dtype())

# ───────────────────────── Data ─────────────────────────
def token_stream(ds_name: str, target: int, seed: int = 42, shuffle_buf: int = 1_000_000):
    ds = load_dataset(ds_name, split="train", streaming=True)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=seed)
    emitted = 0
    for ex in ds:
        enc = tok.encode(ex["text"])
        if EOS is not None and (len(enc) == 0 or enc[-1] != EOS):
            enc = enc + [EOS]
        for t in enc:
            yield t
            emitted += 1
            if emitted >= target:
                return

# ───────────────────────── RelPos (ALiBi) ─────────────────────────
def _alibi_slopes(n_heads: int):
    import math as _m
    def pow2slopes(n):
        start = 2 ** (-2 ** -(_m.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if _m.log2(n_heads).is_integer():
        vals = pow2slopes(n_heads)
    else:
        closest = 2 ** _m.floor(_m.log2(n_heads))
        vals = pow2slopes(closest)
        extra = pow2slopes(2 * closest)
        vals += extra[0::2][: n_heads - closest]
    return torch.tensor(vals, device=DEV).view(1, n_heads, 1, 1)

def alibi_bias(n_heads: int, n_tokens: int):
    i = torch.arange(n_tokens, device=DEV).view(1, 1, n_tokens, 1)
    j = torch.arange(n_tokens, device=DEV).view(1, 1, 1, n_tokens)
    dist = (j - i).clamp_min(0)
    slopes = _alibi_slopes(n_heads)
    return -slopes * dist

# ───────────────────────── Model ─────────────────────────
class LowRankMHA(nn.Module):
    def __init__(self, d: int, h: int, r: int, use_relpos: bool = True):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.use_relpos = use_relpos
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.U = nn.Parameter(torch.randn(self.dk, r))
        nn.init.orthogonal_(self.U)
        self.proj = nn.Linear(h * r, d, bias=False)
        self.drop = nn.Dropout(0.1)

    def _proj(self, x):
        B, N, _ = x.shape
        return (x.view(B, N, self.h, self.dk).transpose(1, 2) @ self.U)

    def forward(self, x, mask=None, rel_bias_tokens: int | None = None):
        q, k, v = self._proj(self.q(x)), self._proj(self.k(x)), self._proj(self.v(x))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        if self.use_relpos and rel_bias_tokens is not None:
            att = att + alibi_bias(self.h, rel_bias_tokens)
        if mask is not None:
            att = att + mask
        z = (att.softmax(-1) @ v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return self.drop(self.proj(z))

class Block(nn.Module):
    def __init__(self, d: int, h: int, r: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = LowRankMHA(d, h, r, use_relpos=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
    def forward(self, x, mask):
        n = x.size(1)
        x = x + self.mha(self.ln1(x), mask, rel_bias_tokens=n)
        return x + self.ff(self.ln2(x))

class Encoder(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        d, l, h, r = cfg["d"], cfg["layers"], cfg["heads"], cfg["rank"]
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d, h, r) for _ in range(l))
        self.ln = nn.LayerNorm(d)
    def forward(self, ids, mask):
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln(x)

class ARHead(nn.Module):
    def __init__(self, d): super().__init__(); self.proj = nn.Linear(d, VOCAB)
    def forward(self, h): return self.proj(h)

class NATHead(nn.Module):
    def __init__(self, d): super().__init__(); self.proj = nn.Linear(d, VOCAB)
    def forward(self, h): return self.proj(h)

class SATHead(nn.Module):
    def __init__(self, d, mode="var"):
        super().__init__()
        self.proj = nn.Linear(d, VOCAB)
        self.mode = mode
        self.gate = nn.Linear(d, 2) if mode == "var" else None
    def forward(self, h_last):
        logits = self.proj(h_last)
        gate = self.gate(h_last[:, 0]) if self.gate is not None else None
        return logits, gate

# ───────────────────────── Masks ─────────────────────────
def causal_mask(n):
    m = torch.full((1, 1, n, n), float("-inf"), device=DEV)
    return torch.triu(m, 1)

def sat_mask(n, block=SAT_BLOCK):
    idx = torch.arange(n, device=DEV)
    grp = idx.unsqueeze(0) // block
    allow = (grp.T == grp) | (grp.T > grp)
    return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

# ───────────────────────── Ckpt I/O ─────────────────────────
def save_ckpt(path: pathlib.Path, core, ar_h, nat_h, sat_h, opt, scaler, meta: Dict[str, Any]):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {
        "core": core.state_dict(),
        "ar": ar_h.state_dict(),
        "nat": nat_h.state_dict(),
        "sat": sat_h.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        **meta,
        "tokenizer_id": TOKENIZER_ID,
    }
    torch.save(state, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)
    (path.parent / "latest.json").write_text(json.dumps({"path": str(path), "step": meta["step"]}))
    print(f"\n✓ saved checkpoint {path.name}")

def load_ckpt(path: pathlib.Path, core, ar_h, nat_h, sat_h, opt, scaler):
    p = _resolve_ckpt(path) or path
    ck = _try_load(p, map_location=DEV)
    if ck is None:
        raise FileNotFoundError(f"No valid checkpoint at {p}")
    core.load_state_dict(ck["core"])
    ar_h.load_state_dict(ck["ar"])
    nat_h.load_state_dict(ck["nat"])
    sat_h.load_state_dict(ck["sat"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    return ck

def _safe_load_any(path: pathlib.Path, tgt: nn.Module, key: str | None = None, rename: str | None = None):
    p = _resolve_ckpt(path) or path
    if not p or not p.exists(): return 0
    ck = _try_load(p, map_location=DEV)
    if ck is None: return 0
    sd = ck.get(key, ck) if key else ck
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if rename:
        sd = {k.replace(rename, "proj."): v for k, v in sd.items() if rename in k}
    tgt_sd = tgt.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt_sd and v.shape == tgt_sd[k].shape}
    if filt:
        tgt.load_state_dict(filt, strict=False)
    return len(filt)

# ───────────────────────── Schedules ─────────────────────────
class WarmupCosine:
    def __init__(self, opt, total_steps, warmup_steps, min_lr_mul=0.1):
        self.opt = opt
        self.total = max(1, total_steps)
        self.warm = max(0, warmup_steps)
        self.min_mul = min_lr_mul
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in opt.param_groups]
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warm:
            t = self.step_num / max(1, self.warm)
            mul = t
        else:
            t = (self.step_num - self.warm) / max(1, self.total - self.warm)
            mul = self.min_mul + 0.5*(1 - self.min_mul)*(1 + math.cos(math.pi * (1 - t)))
        for lr0, g in zip(self.base_lrs, self.opt.param_groups):
            g["lr"] = lr0 * mul

# ───────────────────────── Helpers ─────────────────────────
def count_total_params(mods: List[nn.Module]) -> int:
    return sum(p.numel() for m in mods for p in m.parameters())

def build_core_and_heads(cfg: Dict[str,int]) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    core = Encoder(cfg).to(DEV)
    ar_h = ARHead(cfg["d"]).to(DEV)
    nat_h = NATHead(cfg["d"]).to(DEV)
    sat_h = SATHead(cfg["d"], mode="var").to(DEV)
    return core, ar_h, nat_h, sat_h

def _parse_grow_plan(s: str) -> List[int]:
    steps = []
    for part in s.split(","):
        part = part.strip()
        if part:
            v = int(part)
            if v >= 128:
                steps.append(v)
    return sorted(set(steps))

# ───────────────────────── Train ─────────────────────────
def train(args):
    # Topology resolution:
    # If --resume provided → lock cfg to ckpt['cfg'] (true resume).
    # Else start from preset, optionally doubling with --x2 and/or overriding rank.
    if args.resume:
        ck_head = _try_load(pathlib.Path(args.resume), map_location="cpu")
        if ck_head is None or "cfg" not in ck_head:
            raise RuntimeError(f"Resume requested but checkpoint unusable: {args.resume}")
        cfg = dict(ck_head["cfg"])
        print(f"[resume] locked cfg from ckpt: {cfg}")
    else:
        cfg = PRESETS[args.preset].copy()
        if args.rank: cfg["rank"] = args.rank
        if args.x2:   cfg["layers"] *= 2
        print(f"[fresh/warmstart] cfg: {cfg}")

    BLOCK = args.block or DEFAULT_BLOCK
    batch = max(1, args.batch)
    accum = max(1, args.grad_accum)

    core, ar_h, nat_h, sat_h = build_core_and_heads(cfg)

    # Warm start (shape-safe), optional
    loaded_ws = 0
    if args.warmstart_from:
        src = pathlib.Path(args.warmstart_from)
        loaded_ws += _safe_load_any(src, core, key="core")
        loaded_ws += _safe_load_any(src, ar_h, key="ar")
        loaded_ws += _safe_load_any(src, nat_h, key="nat")
        loaded_ws += _safe_load_any(src, sat_h, key="sat")
        if loaded_ws:
            print(f"Warm-start: loaded {loaded_ws} matching tensors from {src}")

    # Optimizer
    opt = torch.optim.AdamW(
        [
            {"params": core.parameters(),  "lr": args.lr_core},
            {"params": ar_h.parameters(),  "lr": args.lr_head},
            {"params": nat_h.parameters(), "lr": args.lr_head},
            {"params": sat_h.parameters(), "lr": args.lr_head},
        ],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
    )
    scaler = GradScaler(enabled=args.amp and DEV.type == "cuda")

    ce_tok = nn.CrossEntropyLoss(label_smoothing=0.1)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    ce_gate = nn.CrossEntropyLoss()

    # Resume (strict) restores opt/scaler/step/seen_tok
    start_step, seen_tok, last_save_time = 0, 0, time.time()
    if args.resume:
        ck = load_ckpt(pathlib.Path(args.resume), core, ar_h, nat_h, sat_h, opt, scaler)
        start_step = int(ck.get("step", 0))
        seen_tok = int(ck.get("seen_tok", 0))
        last_save_time = float(ck.get("wall_time", time.time()))
        print(f"✓ resumed from step {start_step:,}, seen_tokens={seen_tok:,}")

    # Compute budget
    total_params = count_total_params([core, ar_h, nat_h, sat_h])
    default_target = int(args.ratio * total_params)
    target_tokens = args.target_tokens if args.target_tokens else default_target
    new_tokens_needed = max(0, target_tokens - seen_tok)
    tokens_per_step = batch * BLOCK  # (we use single-seq microbatch, replicated B, then grad-accum)
    eff_tokens_per_update = tokens_per_step * accum

    if new_tokens_needed == 0:
        print("Target already reached – nothing to train.")
        return

    # exact ceil math to hit token budget
    updates_needed = math.ceil(new_tokens_needed / eff_tokens_per_update)
    total_updates = updates_needed
    total_steps = total_updates * accum   # counting optimizer .step() calls as updates
    print(f"[budget] params={total_params:,}  target_tokens={target_tokens:,}  seen={seen_tok:,}")
    print(f"[budget] batch={batch} block={BLOCK} accum={accum}  eff_tokens/update={eff_tokens_per_update}")
    print(f"[budget] updates_needed={updates_needed:,} (ceil), total optimizer steps={updates_needed:,}")

    # Scheduler
    warmup_updates = max(1, int(args.warmup_frac * total_updates))
    sched = WarmupCosine(opt, total_steps=total_updates, warmup_steps=warmup_updates, min_lr_mul=args.min_lr_mul)

    # Data
    stream = token_stream(args.source, target_tokens, seed=42, shuffle_buf=args.shuffle_buf)
    buf: List[int] = []
    pbar = tqdm(total=target_tokens, initial=seen_tok, unit="tok")
    step = start_step
    accum_step = 0
    updates_done = 0
    last_report = time.time()

    # For auto-grow
    grow_plan = _parse_grow_plan(args.grow_plan) if args.auto_grow else []
    if args.auto_grow and BLOCK not in grow_plan:
        grow_plan = sorted(set(grow_plan + [BLOCK]))
    grow_cursor = 0 if not grow_plan else grow_plan.index(BLOCK) if BLOCK in grow_plan else 0
    steps_since_last_grow = 0

    # Training loop
    while updates_done < total_updates:
        # Assemble microbatches
        try:
            while len(buf) < tokens_per_step:
                buf.append(next(stream))
        except StopIteration:
            break

        ids = torch.tensor(buf[:BLOCK], device=DEV).unsqueeze(0)  # 1×N per microbatch; we replicate for batch>1
        buf = buf[BLOCK:]
        # simple data copy for batch>1
        if batch > 1:
            ids = ids.repeat(batch, 1)

        tgt_ar = ids.clone()
        ids_nat = torch.repeat_interleave(ids, 2, 1)

        try:
            with amp(args.amp):
                # AR
                h_ar = core(ids, causal_mask(ids.size(1)))
                logits_ar = ar_h(h_ar)[:, :-1]
                loss_ar = ce_tok(logits_ar.reshape(-1, VOCAB), tgt_ar[:, 1:].reshape(-1)) / accum

                # NAT
                h_nat = core(ids_nat, None)
                log_nat = nat_h(h_nat).log_softmax(-1).transpose(0, 1)  # (T,B,V)
                ilen = tlen = torch.full((ids.size(0),), ids_nat.size(1)//2, device=DEV, dtype=torch.long)
                loss_nat = ctc(log_nat, tgt_ar, ilen, tlen) / accum

                # SAT
                h_sat = core(ids, sat_mask(ids.size(1)))
                logits_sat, gate = sat_h(h_sat[:, -SAT_BLOCK:])
                tgt_sat = ids[:, 1:SAT_BLOCK+1]
                ls = ce_tok(logits_sat.reshape(-1, VOCAB), tgt_sat.reshape(-1)) / accum
                if gate is not None:
                    ls += (EMIT_LAMBDA * ce_gate(gate, torch.ones(ids.size(0), device=DEV, dtype=torch.long))) / accum

                loss = loss_ar + loss_nat + ls

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step == accum:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                updates_done += 1
                sched.step()
                accum_step = 0

                seen_tok += eff_tokens_per_update
                pbar.update(min(eff_tokens_per_update, max(0, target_tokens - pbar.n)))

                # checkpoint cadence
                now = time.time()
                time_due = (now - last_save_time) >= args.save_every_sec > 0
                step_due = args.save_every_steps > 0 and updates_done % args.save_every_steps == 0
                if time_due or step_due:
                    ck_name = f"step{(start_step + updates_done):08d}.pt"
                    save_ckpt(
                        pathlib.Path(args.save_dir) / ck_name,
                        core, ar_h, nat_h, sat_h, opt, scaler,
                        meta={
                            "cfg": cfg,
                            "step": start_step + updates_done,
                            "seen_tok": seen_tok,
                            "wall_time": now,
                            "py_state": random.getstate(),
                            "torch_state": rng_state(),
                        },
                    )
                    last_save_time = now

                # auto-grow
                if args.auto_grow:
                    steps_since_last_grow += 1
                    if steps_since_last_grow >= args.grow_every_updates:
                        steps_since_last_grow = 0
                        if grow_plan and grow_cursor + 1 < len(grow_plan):
                            candidate = grow_plan[grow_cursor + 1]
                            print(f"[auto-grow] attempting BLOCK {BLOCK} -> {candidate}")
                            BLOCK = candidate
                            grow_cursor += 1
                            tokens_per_step = batch * BLOCK
                            eff_tokens_per_update = tokens_per_step * accum
                            torch.cuda.empty_cache()

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                # Backoff priority: block → batch → accum
                changed = False
                if BLOCK > 128:
                    new_block = max(128, BLOCK // 2)
                    print(f"\n[OOM] reducing block from {BLOCK} -> {new_block}")
                    BLOCK = new_block
                    changed = True
                elif batch > 1:
                    nb = max(1, batch // 2)
                    if nb < batch:
                        print(f"\n[OOM] reducing batch from {batch} -> {nb}")
                        batch = nb
                        changed = True
                elif accum > 1:
                    na = max(1, accum // 2)
                    if na < accum:
                        print(f"\n[OOM] reducing grad_accum from {accum} -> {na}")
                        accum = na
                        changed = True
                if changed:
                    tokens_per_step = batch * BLOCK
                    eff_tokens_per_update = tokens_per_step * accum
                    torch.cuda.empty_cache()
                    continue
            raise

        # progress printing throttle
        step += 1
        if time.time() - last_report >= 5:
            last_report = time.time()
            pbar.set_postfix(
                block=BLOCK, batch=batch, accum=accum,
                updates=f"{updates_done}/{total_updates}"
            )

    pbar.close()
    save_ckpt(
        pathlib.Path(args.save_dir) / "final.pt",
        core, ar_h, nat_h, sat_h, opt, scaler,
        meta={
            "cfg": cfg,
            "step": start_step + updates_done,
            "seen_tok": seen_tok,
            "wall_time": time.time(),
            "py_state": random.getstate(),
            "torch_state": rng_state(),
        },
    )
    print("🎉 training complete")

# ───────────────────────── Inference ─────────────────────────
def load_joint(ckpt: str):
    path = _resolve_ckpt(pathlib.Path(ckpt)) or pathlib.Path(ckpt)
    sd = _try_load(path, map_location=DEV)
    if sd is None:
        raise FileNotFoundError(f"No valid checkpoint at {path}")
    cfg = sd["cfg"]
    core, ar_h, nat_h, sat_h = build_core_and_heads(cfg)
    core.load_state_dict(sd["core"])
    ar_h.load_state_dict(sd["ar"])
    nat_h.load_state_dict(sd["nat"])
    sat_h.load_state_dict(sd["sat"])
    return core, ar_h, nat_h, sat_h

@torch.no_grad()
def ar_decode(core, ar_h, prompt: str, max_new: int, T: float):
    ids = torch.tensor([tok.encode(prompt)], device=DEV)
    t0 = time.time()
    for _ in range(max_new):
        h = core(ids, causal_mask(ids.size(1)))
        nxt = (ar_h(h)[:, -1] / max(T, 1e-5)).softmax(-1).multinomial(1)
        ids = torch.cat([ids, nxt], 1)
    print(tok.decode(ids[0].tolist(), skip_special_tokens=True))
    print(f"[{max_new} tok in {time.time() - t0:.2f}s]")

@torch.no_grad()
def sat_decode(core, sat_h, prompt, max_new, T, var):
    ids = torch.tensor([tok.encode(prompt)], device=DEV)
    added, t0 = 0, time.time()
    while added < max_new:
        h = core(ids, sat_mask(ids.size(1)))
        logits, gate = sat_h(h[:, -SAT_BLOCK:])
        stride = 2 if (not var or gate is None) else (gate.softmax(-1).multinomial(1) + 1).item()
        probs = torch.softmax(logits / T, -1)[:, :stride]
        nxt = probs.reshape(1, stride, VOCAB).multinomial(1).squeeze(-1)
        ids = torch.cat([ids, nxt], 1)
        added += stride
    print(tok.decode(ids[0].tolist(), skip_special_tokens=True))
    print(f"[{added} tok in {time.time() - t0:.2f}s]")

@torch.no_grad()
def nat_decode(core, nat_h, prompt, max_new, passes, streams):
    ids = torch.tensor([tok.encode(prompt) + [BLANK] * (max_new * 2)], device=DEV)
    t0 = time.time()
    for _ in range(passes):
        h = core(ids, None)
        logits = nat_h(h)
        logits[..., BLANK] = -1e9
        cand = logits.topk(streams, -1).indices.permute(2, 0, 1)
        best = (cand != BLANK).float().mean(-1).argmax(0)
        ids = cand[best, torch.arange(ids.size(0), device=DEV)][:, ::2]
    out = [t for t in ids[0].tolist() if t != BLANK]
    print(tok.decode(out, skip_special_tokens=True))
    print(f"[{len(out)} output tokens in {time.time() - t0:.2f}s]")

# ───────────────────────── CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--preset", choices=PRESETS, default="small")
    tr.add_argument("--rank", type=int)
    tr.add_argument("--x2", action="store_true", help="Double layers from preset (ignored when --resume locks cfg)")
    tr.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    tr.add_argument("--batch", type=int, default=1, help="microbatch size")
    tr.add_argument("--grad_accum", type=int, default=1, help="optimizer steps after this many microbatches")
    tr.add_argument("--source", default="cerebras/SlimPajama-627B")
    tr.add_argument("--ratio", type=float, default=25.0, help="Chinchilla tokens/param (default 25× total params)")
    tr.add_argument("--target_tokens", type=int, help="override computed budget")
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--save_every_sec", type=int, default=8*24*3600)
    tr.add_argument("--save_every_steps", type=int, default=0)
    tr.add_argument("--save_dir", default=str(CKDIR))
    tr.add_argument("--resume", type=str, help="strict resume; locks cfg to ckpt")
    tr.add_argument("--warmstart_from", type=str, help="shape-safe init from ckpt (topology may differ)")
    tr.add_argument("--auto_grow", action="store_true")
    tr.add_argument("--grow_plan", type=str, default="576,640,768,896,1024")
    tr.add_argument("--grow_every_updates", type=int, default=2000)
    tr.add_argument("--warmup_frac", type=float, default=0.02)
    tr.add_argument("--min_lr_mul", type=float, default=0.15)
    tr.add_argument("--lr_core", type=float, default=5e-5)
    tr.add_argument("--lr_head", type=float, default=2e-4)
    tr.add_argument("--shuffle_buf", type=int, default=1_000_000)

    inf = sub.add_parser("infer")
    inf.add_argument("--mode", choices=["ar", "nat", "sat"], required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=120)
    inf.add_argument("--temperature", type=float, default=1.0)
    inf.add_argument("--var", action="store_true")
    inf.add_argument("--passes", type=int, default=1)
    inf.add_argument("--streams", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        core, ar_h, nat_h, sat_h = load_joint(args.ckpt)
        if args.mode == "ar":
            ar_decode(core, ar_h, args.prompt, args.max_new, args.temperature)
        elif args.mode == "sat":
            sat_decode(core, sat_h, args.prompt, args.max_new, args.temperature, args.var)
        else:
            nat_decode(core, nat_h, args.prompt, args.max_new, args.passes, args.streams)

if __name__ == "__main__":
    main()
