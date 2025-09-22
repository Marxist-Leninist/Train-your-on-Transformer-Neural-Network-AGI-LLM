#!/usr/bin/env python3
# 5L.py — joint AR+NAT+SAT trainer/decoder (Qwen3 tokenizer)
# Robust fresh-start, ignores *.pt.tmp, AMP dtype auto, OOM backoff, progressive block growth.
# Added: repetition/presence/frequency penalties, top-k/top-p/min-p, greedy, no-repeat-ngrams.
# Fixes: SAT multinomial shape; checkpoint loads on CPU; cfg fallback if ckpt missing cfg.
# UPDATE: time-based checkpointing only (monotonic), no step-based saving. Resume respects interval.

from __future__ import annotations
import argparse, json, math, pathlib, random, time, os
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple

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

# Use the Qwen3 tokenizer (can override with env TOKENIZER_ID if needed)
TOKENIZER_ID = os.environ.get(
    "TOKENIZER_ID",
    "Qwen/Qwen3-235B-A22B-Thinking-2507"
)

# Some Qwen tokenizers require trust_remote_code
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})
VOCAB, BLANK, EOS = (
    max(tok.get_vocab().values()) + 1,   # allow new [PAD] if appended
    tok.pad_token_id,
    tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
)

PRESETS: Dict[str, Dict[str, int]] = {
    "small":   dict(d=512, layers=8,  heads=16, rank=64),
    "smallx2": dict(d=512, layers=16, heads=16, rank=64),
    "base":    dict(d=768, layers=12, heads=24, rank=96),
}

# Safe default for 1× Tesla P40; override with --block
DEFAULT_BLOCK = 576
SAT_BLOCK = 2
LR_CORE, LR_HEAD = 5e-5, 2e-4
EMIT_LAMBDA = 0.1
# Default interval: 24 hours. Override with --save_every_sec (e.g., 86400).
DEFAULT_SAVE_SEC = 24 * 3600
CKDIR = pathlib.Path("ckpts_joint")


# ───────────────────────── Utilities ─────────────────────────
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
    """
    Return a solid .pt (never .tmp). If 'path' is dir, pick newest *.pt.
    If not usable, return None.
    """
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
    """
    Always load on CPU to avoid CUDA fragmentation/OOM during torch.load.
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ckpt-skip] {path} not usable: {e}")
        return None


# ───────────────────────── AMP helper ─────────────────────────
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _auto_amp_dtype():
    if DEV.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        except Exception:
            return torch.float16
    return torch.float32

def amp(enabled: bool):
    # Only enable if explicitly requested AND CUDA is available
    return nullcontext() if not (enabled and DEV.type == "cuda") else _ac(device_type="cuda", dtype=_auto_amp_dtype())


# ───────────────────────── Data stream ─────────────────────────
def token_stream(ds_name: str, target: int, seed: int = 42):
    ds = load_dataset(ds_name, split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10_000, seed=seed)
    emitted = 0
    for ex in ds:
        # ensure EOS between docs
        enc = tok.encode(ex["text"])
        if EOS is not None and (len(enc) == 0 or enc[-1] != EOS):
            enc = enc + [EOS]
        for t in enc:
            yield t
            emitted += 1
            if emitted >= target:
                return


# ───────────────────────── Relative positional bias (ALiBi) ─────────────────────────
def _alibi_slopes(n_heads: int):
    import math
    def pow2slopes(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        vals = pow2slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        vals = pow2slopes(closest)
        extra = pow2slopes(2 * closest)
        vals += extra[0::2][: n_heads - closest]
    return torch.tensor(vals, device=DEV).view(1, n_heads, 1, 1)

def alibi_bias(n_heads: int, n_tokens: int):
    i = torch.arange(n_tokens, device=DEV).view(1, 1, n_tokens, 1)
    j = torch.arange(n_tokens, device=DEV).view(1, 1, 1, n_tokens)
    dist = (j - i).clamp_min(0)  # only penalize future
    slopes = _alibi_slopes(n_heads)
    return -slopes * dist


# ───────────────────────── Model components ─────────────────────────
class LowRankMHA(nn.Module):
    """
    Cache-aware MHA with low-rank projections; supports kv caching for decode.
    """
    def __init__(self, d: int, h: int, r: int, use_relpos: bool = True):
        super().__init__()
        assert d % h == 0, "d must be divisible by number of heads"
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rel_bias_tokens: Optional[int] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        q = self._proj(self.q(x))
        k_new = self._proj(self.k(x))
        v_new = self._proj(self.v(x))

        if kv_cache is None:
            k, v = k_new, v_new
        else:
            k, v = kv_cache
            if use_cache:
                k = torch.cat([k, k_new], dim=2)
                v = torch.cat([v, v_new], dim=2)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)

        if q.size(2) == k.size(2):
            if self.use_relpos and rel_bias_tokens is not None:
                att = att + alibi_bias(self.h, rel_bias_tokens)
            if mask is not None:
                att = att + mask

        z = (att.softmax(-1) @ v).transpose(1, 2)  # (B,Nq,h,r)
        z = z.reshape(x.size(0), x.size(1), -1)
        out = self.drop(self.proj(z))
        return (out, (k, v)) if use_cache else out


class Block(nn.Module):
    def __init__(self, d: int, h: int, r: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = LowRankMHA(d, h, r, use_relpos=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        n = x.size(1)
        if use_cache:
            y, new_kv = self.mha(self.ln1(x), mask, rel_bias_tokens=n if mask is not None else None, kv_cache=kv, use_cache=True)
            x = x + y
            x = x + self.ff(self.ln2(x))
            return x, new_kv
        else:
            x = x + self.mha(self.ln1(x), mask, rel_bias_tokens=n)
            return x + self.ff(self.ln2(x))


class Encoder(nn.Module):
    """
    Transformer encoder with optional kv caching (for AR/SAT decode).
    """
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()
        d, l, h, r = cfg["d"], cfg["layers"], cfg["heads"], cfg["rank"]
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, h, r) for _ in range(l)])
        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        ids: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False
    ):
        x = self.emb(ids)
        if not use_cache:
            for blk in self.blocks:
                x = blk(x, mask)
            return self.ln(x)

        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if (kv_caches is not None) else None
            x, kv_out = blk(x, mask, kv, use_cache=True)
            new_kvs.append(kv_out)
        return self.ln(x), new_kvs


class ARHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, VOCAB)
    def forward(self, h): return self.proj(h)


class NATHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, VOCAB)
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


# ───────────────────────── Checkpoint helpers ─────────────────────────
def save_ckpt(
    path: pathlib.Path,
    core: nn.Module,
    ar_h: nn.Module,
    nat_h: nn.Module,
    sat_h: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    meta: Dict[str, Any],
):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {
        "core": core.state_dict(),
        "ar": ar_h.state_dict(),
        "nat": nat_h.state_dict(),
        "sat": sat_h.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": meta.get("cfg"),
        "tokenizer_id": TOKENIZER_ID,
        **{k: v for k, v in meta.items() if k not in {"cfg"}},
    }
    torch.save(state, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)
    (path.parent / "latest.json").write_text(json.dumps({"path": str(path), "step": meta["step"]}))
    print(f"\n✓ saved checkpoint {path.name}")

def load_ckpt(
    path: pathlib.Path,
    core: nn.Module,
    ar_h: nn.Module,
    nat_h: nn.Module,
    sat_h: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
):
    p = _resolve_ckpt(path) or path
    ck = _try_load(p, map_location="cpu")
    if ck is None:
        raise FileNotFoundError(f"No valid checkpoint at {p}")
    core.load_state_dict(ck["core"])
    ar_h.load_state_dict(ck["ar"])
    nat_h.load_state_dict(ck["nat"])
    sat_h.load_state_dict(ck["sat"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("seen_tok", 0), ck.get("wall_time", time.time())

def _safe_load_any(path: pathlib.Path, tgt: nn.Module, key: str | None = None, rename: str | None = None):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return 0
    ck = _try_load(p, map_location="cpu")
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

def infer_cfg_from_ckpt(path: pathlib.Path):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return None
    sd = _try_load(p, map_location="cpu")
    if sd is None: return None
    if isinstance(sd, dict) and "cfg" in sd and isinstance(sd["cfg"], dict):
        return dict(sd["cfg"])
    core = sd.get("core")
    if core is None: return None
    emb_w = core.get("emb.weight")
    if emb_w is None: return None
    d = emb_w.shape[1]
    layer_ids = []
    for k in core.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                layer_ids.append(int(parts[1]))
    layers = (max(layer_ids) + 1) if layer_ids else None
    U = core.get("blocks.0.mha.U")
    heads = rank = None
    if U is not None:
        dk, r = U.shape
        rank = r
        heads = d // dk if dk > 0 else None
    out = {"d": d}
    if layers is not None: out["layers"] = layers
    if heads is not None:  out["heads"] = heads
    if rank is not None:   out["rank"] = rank
    return out


# ───────────────────────── Train loop ─────────────────────────
def _parse_grow_plan(s: str) -> List[int]:
    steps = []
    for part in s.split(","):
        part = part.strip()
        if part:
            v = int(part)
            if v >= 128:
                steps.append(v)
    return sorted(set(steps))

def _init_save_timers(resume_wall_time: float | None, interval_sec: int) -> Tuple[float, float]:
    """
    Returns (last_save_wall, last_save_mono).
    We use wall time for metadata, monotonic for interval checks.
    If resuming and the last save was long ago, schedule next save accordingly.
    """
    now_wall = time.time()
    now_mono = time.monotonic()
    if resume_wall_time is None:
        return now_wall, now_mono
    # How long since the previous save in wall-clock?
    elapsed_wall = max(0.0, now_wall - resume_wall_time)
    # Clamp to interval so we don't try to "catch up" multiple times
    elapsed_clamped = min(float(interval_sec), elapsed_wall)
    # Pretend we last saved 'elapsed_clamped' ago on the monotonic clock
    return now_wall, now_mono - elapsed_clamped

def train(args):
    cfg = PRESETS[args.preset].copy()

    # Previous topology probe (unless --fresh)
    if not args.fresh:
        src_probe = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        prev_cfg = infer_cfg_from_ckpt(src_probe)
    else:
        prev_cfg = None

    if prev_cfg:
        cfg["d"] = prev_cfg.get("d", cfg["d"])
        if prev_cfg.get("heads"):
            cfg["heads"] = prev_cfg["heads"]
        if args.rank is None and prev_cfg.get("rank"):
            cfg["rank"] = prev_cfg["rank"]
        if prev_cfg.get("layers"):
            cfg["layers"] = prev_cfg["layers"]
        if args.x2 and prev_cfg.get("layers"):
            cfg["layers"] = max(cfg["layers"], prev_cfg["layers"] * 2)
    if args.rank:
        cfg["rank"] = args.rank
    if args.x2 and not prev_cfg:
        cfg["layers"] *= 2

    BLOCK = args.block or DEFAULT_BLOCK

    core = Encoder(cfg).to(DEV)
    ar_h, nat_h = ARHead(cfg["d"]).to(DEV), NATHead(cfg["d"]).to(DEV)
    sat_h = SATHead(cfg["d"], mode="var").to(DEV)

    # Warm start unless --fresh
    loaded = 0
    if not args.fresh:
        src = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        src = _resolve_ckpt(src)
        if src:
            loaded += _safe_load_any(src, core, key="core")
            loaded += _safe_load_any(src, ar_h, key="ar")
            loaded += _safe_load_any(src, nat_h, key="nat")
            loaded += _safe_load_any(src, sat_h, key="sat")
            if loaded:
                print(f"Warm-start: loaded {loaded} matching tensors from {src}")

    opt = torch.optim.AdamW(
        [
            {"params": core.parameters(), "lr": LR_CORE},
            {"params": ar_h.parameters(), "lr": LR_HEAD},
            {"params": nat_h.parameters(), "lr": LR_HEAD},
            {"params": sat_h.parameters(), "lr": LR_HEAD},
        ]
    )
    scaler = GradScaler(enabled=(args.amp and DEV.type == "cuda"))

    ce_tok = nn.CrossEntropyLoss(label_smoothing=0.1)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    ce_gate = nn.CrossEntropyLoss()

    # ---------- resume bookkeeping ----------
    start_step, seen_tok = 0, 0
    last_save_wall = None
    if args.resume and not args.fresh:
        start_step, seen_tok, last_save_wall = load_ckpt(
            pathlib.Path(args.resume), core, ar_h, nat_h, sat_h, opt, scaler
        )
        print(f"✓ resumed from step {start_step:,}, seen_tokens={seen_tok:,}")
    # Initialize save timers
    last_save_wall, last_save_mono = _init_save_timers(last_save_wall, args.save_every_sec)

    # Target tokens
    if args.target_tokens:
        target_tokens = args.target_tokens
    else:
        param_count = sum(p.numel() for p in core.parameters())
        target_tokens = int(25 * param_count)

    new_tokens_needed = target_tokens - seen_tok
    if new_tokens_needed <= 0:
        print("Target already reached – nothing to train.")
        return
    new_steps = new_tokens_needed // BLOCK
    if args.steps:
        new_steps = min(new_steps, args.steps)
        new_tokens_needed = new_steps * BLOCK

    total_tokens_needed = seen_tok + new_tokens_needed
    print(f"[auto-steps] {new_steps:,} training steps (@ {BLOCK} tokens/step)")

    # Progressive growth plan
    grow_plan = _parse_grow_plan(args.grow_plan) if args.auto_grow else []
    if args.auto_grow:
        if BLOCK not in grow_plan:
            grow_plan = sorted(set(grow_plan + [BLOCK]))
        print(f"[auto-grow] plan: {grow_plan} every {args.grow_every_steps} steps")

    stream = token_stream(args.source, target_tokens, seed=42)
    buf: list[int] = []
    pbar = tqdm(total=total_tokens_needed, initial=seen_tok, unit="tok")
    step = start_step
    steps_since_last_grow = 0

    while seen_tok < total_tokens_needed:
        # ------- assemble one batch -------
        try:
            while len(buf) < BLOCK:
                buf.append(next(stream))
        except StopIteration:
            break
        ids = torch.tensor(buf[:BLOCK], device=DEV).unsqueeze(0)  # (B=1, N)
        buf = buf[BLOCK:]

        tgt_ar = ids.clone()                           # (1, N)
        ids_nat = torch.repeat_interleave(ids, 2, 1)   # (1, 2N) for NAT only

        try:
            with amp(args.amp):
                # AR path
                h_ar = core(ids, causal_mask(ids.size(1)))
                logits_ar = ar_h(h_ar)[:, :-1]
                loss_ar = ce_tok(logits_ar.reshape(-1, VOCAB), tgt_ar[:, 1:].reshape(-1))

                # NAT path (uses doubled sequence)
                h_nat = core(ids_nat, None)
                log_nat = nat_h(h_nat).log_softmax(-1).transpose(0, 1)  # (T,B,V)
                ilen = tlen = torch.tensor([ids_nat.size(1) // 2], device=DEV)
                loss_nat = ctc(log_nat, tgt_ar, ilen, tlen)

                # SAT path
                h_sat = core(ids, sat_mask(ids.size(1)))
                logits_sat, gate = sat_h(h_sat[:, -SAT_BLOCK:])
                tgt_sat = ids[:, 1:SAT_BLOCK+1]
                loss_sat = ce_tok(logits_sat.reshape(-1, VOCAB), tgt_sat.reshape(-1))
                if gate is not None:
                    loss_sat += EMIT_LAMBDA * ce_gate(gate, torch.ones(ids.size(0), device=DEV, dtype=torch.long))

                loss = loss_ar + loss_nat + loss_sat

            # optimisation
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                new_block = max(128, BLOCK // 2)
                if new_block < BLOCK:
                    print(f"\n[OOM] reducing block from {BLOCK} -> {new_block}")
                    BLOCK = new_block
                    if DEV.type == "cuda":
                        torch.cuda.empty_cache()
                    buf = ids[0].tolist() + buf
                    steps_since_last_grow = 0
                    continue
            raise

        # progress
        step += 1
        seen_tok += BLOCK
        pbar.update(BLOCK)
        pbar.set_postfix(loss=f"{loss.item():.3f}", block=BLOCK)

        # time-based checkpoint cadence only (monotonic)
        if args.save_every_sec > 0:
            now_mono = time.monotonic()
            if now_mono - last_save_mono >= args.save_every_sec:
                ck_name = f"step{step:08d}.pt"
                save_ckpt(
                    pathlib.Path(args.save_dir) / ck_name,
                    core, ar_h, nat_h, sat_h, opt, scaler,
                    meta={
                        "cfg": cfg,
                        "step": step,
                        "seen_tok": seen_tok,
                        "wall_time": time.time(),
                        "py_state": random.getstate(),
                        "torch_state": rng_state(),
                    },
                )
                last_save_mono = now_mono
                last_save_wall = time.time()

        # progressive growth
        if args.auto_grow:
            steps_since_last_grow += 1
            if steps_since_last_grow >= args.grow_every_steps:
                steps_since_last_grow = 0
                try:
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        candidate = grow_plan[idx + 1]
                        print(f"[auto-grow] attempting BLOCK {BLOCK} -> {candidate}")
                        BLOCK = candidate
                        if DEV.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        print("[auto-grow] at max planned block; no further growth.")
                except ValueError:
                    grow_plan = sorted(set(grow_plan + [BLOCK]))
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        candidate = grow_plan[idx + 1]
                        print(f"[auto-grow] moving to planned BLOCK {candidate}")
                        BLOCK = candidate
                        if DEV.type == "cuda":
                            torch.cuda.empty_cache()

    pbar.close()

    # final save
    save_ckpt(
        pathlib.Path(args.save_dir) / "final.pt",
        core, ar_h, nat_h, sat_h, opt, scaler,
        meta={
            "cfg": cfg,
            "step": step,
            "seen_tok": seen_tok,
            "wall_time": time.time(),
            "py_state": random.getstate(),
            "torch_state": rng_state(),
        },
    )
    print("🎉 training complete")


# ───────────────────────── Sampling utils ─────────────────────────
def _apply_no_repeat_ngram(logits: torch.Tensor, ids: torch.Tensor, n: int):
    """
    Block tokens that would complete any previously seen n-gram.
    ids: (1, t)
    logits: (..., V) where ... may be (1,) or (stride,)
    """
    if n <= 0 or ids.size(1) < n - 1:
        return logits
    prefix = ids[0, - (n - 1):].tolist()
    # Build set of next tokens forbidden after this prefix.
    banned = []
    tokens = ids[0].tolist()
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n - 1] == prefix:
            banned.append(tokens[i + n - 1])
    if banned:
        banned_idx = torch.tensor(banned, device=logits.device, dtype=torch.long)
        logits[..., banned_idx] = float("-inf")
    return logits


def _apply_rep_presence_frequency(
    logits: torch.Tensor, ids: torch.Tensor, last_n: int,
    repetition_penalty: float, presence_penalty: float, frequency_penalty: float
):
    """
    logits: (..., V) where ... may be (1,) or (stride,)
    ids: (1, t) history
    """
    if ids.numel() == 0:
        return logits
    if last_n > 0:
        hist = ids[0, -last_n:].to(torch.long)
    else:
        hist = ids[0].to(torch.long)

    if hist.numel() == 0:
        return logits

    uniq, counts = torch.unique(hist, return_counts=True)

    # presence/frequency penalties (OpenAI-like)
    if presence_penalty != 0.0 or frequency_penalty != 0.0:
        # subtract presence for seen tokens; subtract frequency * count
        adjust = presence_penalty + frequency_penalty * counts.to(logits.dtype)
        logits[..., uniq] = logits[..., uniq] - adjust

    # repetition penalty (CTRL/GPT-NeoX style)
    if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6:
        sel = logits[..., uniq]
        # if logit > 0: divide by penalty; else multiply by penalty
        sel = torch.where(sel > 0, sel / repetition_penalty, sel * repetition_penalty)
        logits[..., uniq] = sel

    return logits


def _filter_top_k_top_p_min_p(
    logits: torch.Tensor, top_k: int, top_p: float, min_p: float, temperature: float
) -> torch.Tensor:
    """
    Works on 1D or 2D logits (..., V). Applies temperature, then filtering.
    Returns normalized probabilities ready for sampling.
    """
    logits = logits / max(temperature, 1e-8)

    # shape handling
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    B, V = logits.size(0), logits.size(-1)

    # Convert to probabilities for p-based filtering
    probs = logits.softmax(-1)

    # Top-k
    if top_k and top_k < V:
        vals, idx = torch.topk(probs, top_k, dim=-1)
        mask = torch.full_like(probs, 0.0)
        mask.scatter_(1, idx, 1.0)
        probs = probs * mask

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        # Always keep at least one
        keep[..., 0] = True
        # Build mask
        mask = torch.zeros_like(probs)
        mask.scatter_(1, sorted_idx, keep.to(mask.dtype))
        probs = probs * mask

    # Min-p
    if min_p > 0.0:
        probs = torch.where(probs >= min_p, probs, torch.zeros_like(probs))

    # If everything zeroed (can happen at extreme settings), fall back to the argmax token
    sums = probs.sum(-1, keepdim=True)
    empty = (sums == 0)
    if empty.any():
        fallback_idx = logits.argmax(-1, keepdim=True)
        probs = torch.where(empty, torch.zeros_like(probs), probs)
        probs.scatter_(-1, fallback_idx, torch.where(empty, torch.ones_like(sums), torch.zeros_like(sums)))

    # Renormalize
    probs = probs / probs.sum(-1, keepdim=True)
    return probs


# ───────────────────────── Inference helpers ─────────────────────────
def load_joint(ckpt: str, preset: str):
    path = _resolve_ckpt(pathlib.Path(ckpt)) or pathlib.Path(ckpt)
    sd = _try_load(path, map_location="cpu")
    if sd is None:
        raise FileNotFoundError(f"No valid checkpoint at {path}")
    cfg = sd["cfg"] if "cfg" in sd and isinstance(sd["cfg"], dict) else (infer_cfg_from_ckpt(path) or PRESETS[preset])
    core = Encoder(cfg).to(DEV)
    ar_h, nat_h = ARHead(cfg["d"]).to(DEV), NATHead(cfg["d"]).to(DEV)
    sat_h = SATHead(cfg["d"]).to(DEV)
    core.load_state_dict(sd["core"])
    ar_h.load_state_dict(sd["ar"])
    nat_h.load_state_dict(sd["nat"])
    sat_h.load_state_dict(sd["sat"])
    return core, ar_h, nat_h, sat_h


@torch.no_grad()
def ar_decode(core, ar_h, prompt: str, max_new: int, T: float,
              greedy: bool, top_k: int, top_p: float, min_p: float,
              repetition_penalty: float, presence_penalty: float,
              frequency_penalty: float, penalty_last_n: int,
              no_repeat_ngram_size: int):
    ids = torch.tensor([tok.encode(prompt)], device=DEV)
    if ids.size(1) == 0:
        ids = torch.tensor([[EOS] if EOS is not None else [0]], device=DEV)
    h_full, kvs = core(ids, causal_mask(ids.size(1)), use_cache=True)

    start = time.time()
    for _ in range(max_new):
        logits = ar_h(h_full)[:, -1]  # (1, V)

        # penalties
        logits = _apply_no_repeat_ngram(logits, ids, no_repeat_ngram_size)
        logits = _apply_rep_presence_frequency(
            logits, ids, penalty_last_n, repetition_penalty, presence_penalty, frequency_penalty
        )

        if greedy:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = _filter_top_k_top_p_min_p(logits.squeeze(0), top_k, top_p, min_p, T)
            nxt = probs.multinomial(1)

        ids = torch.cat([ids, nxt.unsqueeze(0) if nxt.dim()==1 else nxt], 1)

        # step with kv cache
        x = ids[:, -1:]
        h_full, kvs = core(x, None, kv_caches=kvs, use_cache=True)

    print(tok.decode(ids[0].tolist(), skip_special_tokens=True))
    print(f"[{max_new} tok in {time.time() - start:.2f}s]")


@torch.no_grad()
def sat_decode(core, sat_h, prompt, max_new, T, var,
               greedy: bool, top_k: int, top_p: float, min_p: float,
               repetition_penalty: float, presence_penalty: float,
               frequency_penalty: float, penalty_last_n: int,
               no_repeat_ngram_size: int):
    ids = torch.tensor([tok.encode(prompt)], device=DEV)
    added, t0 = 0, time.time()
    while added < max_new:
        h = core(ids, sat_mask(ids.size(1)))
        logits_all, gate = sat_h(h[:, -SAT_BLOCK:])  # (1, SAT_BLOCK, V)
        stride = 2 if (not var or gate is None) else (gate.softmax(-1).multinomial(1) + 1).item()
        stride = int(stride)

        # Sequentially sample within the stride so penalties apply cumulatively
        for pos in range(stride):
            row_logits = logits_all[:, pos, :]  # (1, V)

            # penalties
            row_logits = _apply_no_repeat_ngram(row_logits, ids, no_repeat_ngram_size)
            row_logits = _apply_rep_presence_frequency(
                row_logits, ids, penalty_last_n, repetition_penalty, presence_penalty, frequency_penalty
            )

            if greedy:
                nxt = row_logits.argmax(-1, keepdim=True)  # (1,1)
            else:
                probs = _filter_top_k_top_p_min_p(row_logits.squeeze(0), top_k, top_p, min_p, T)
                nxt = probs.multinomial(1)  # (1,1)

            ids = torch.cat([ids, nxt], 1)
            added += 1
            if added >= max_new:
                break

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
    tr.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    tr.add_argument("--source", default="cerebras/SlimPajama-627B")
    tr.add_argument("--target_tokens", type=int)
    tr.add_argument("--steps", type=int)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--save_every_sec", type=int, default=DEFAULT_SAVE_SEC)
    tr.add_argument("--save_dir", default=str(CKDIR))
    tr.add_argument("--resume", type=str)
    tr.add_argument("--x2", action="store_true", help="~2x params by doubling layers")
    tr.add_argument("--warmstart_from", type=str, default=None, help="Path to previous final.pt for shape-safe warm start")
    tr.add_argument("--fresh", action="store_true", help="Start from scratch: do not probe or load any checkpoints")

    # Progressive block growth
    tr.add_argument("--auto_grow", action="store_true", help="Automatically grow block size over time")
    tr.add_argument("--grow_plan", type=str, default="576,640,768,896,1024", help="Comma list of block sizes to try in order")
    tr.add_argument("--grow_every_steps", type=int, default=50000, help="Steps between growth attempts")

    inf = sub.add_parser("infer")
    inf.add_argument("--mode", choices=["ar", "nat", "sat"], required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--preset", default="small")
    inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=120)
    inf.add_argument("--temperature", type=float, default=1.0)

    # New decode controls
    inf.add_argument("--greedy", action="store_true", help="Greedy decode (overrides sampling)")
    inf.add_argument("--top_k", type=int, default=0)
    inf.add_argument("--top_p", type=float, default=1.0)
    inf.add_argument("--min_p", type=float, default=0.0)

    inf.add_argument("--repetition_penalty", type=float, default=1.0)
    inf.add_argument("--presence_penalty", type=float, default=0.0)
    inf.add_argument("--frequency_penalty", type=float, default=0.0)
    inf.add_argument("--penalty_last_n", type=int, default=64)
    inf.add_argument("--no_repeat_ngram_size", type=int, default=0)

    inf.add_argument("--var", action="store_true")
    inf.add_argument("--passes", type=int, default=1)
    inf.add_argument("--streams", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        core, ar_h, nat_h, sat_h = load_joint(args.ckpt, args.preset)
        if args.mode == "ar":
            ar_decode(core, ar_h, args.prompt, args.max_new, args.temperature,
                      args.greedy, args.top_k, args.top_p, args.min_p,
                      args.repetition_penalty, args.presence_penalty,
                      args.frequency_penalty, args.penalty_last_n,
                      args.no_repeat_ngram_size)
        elif args.mode == "sat":
            sat_decode(core, sat_h, args.prompt, args.max_new, args.temperature, args.var,
                       args.greedy, args.top_k, args.top_p, args.min_p,
                       args.repetition_penalty, args.presence_penalty,
                       args.frequency_penalty, args.penalty_last_n,
                       args.no_repeat_ngram_size)
        else:
            nat_decode(core, nat_h, args.prompt, args.max_new, args.passes, args.streams)


if __name__ == "__main__":
    main()
