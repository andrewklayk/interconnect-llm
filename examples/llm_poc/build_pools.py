"""Phase A of the LLM ergodicity PoC: precompute the conditional sentiment law.

Because the closed loop is Markov in the instruction level ``k`` and responses are
i.i.d. given ``k``, the *only* thing the loop dynamics depend on is the
conditional law of the sentiment observation given ``k``.  Sampling that law once
per level and reusing it makes every subsequent trajectory essentially free, so
the total LLM cost is ``num_levels x samples_per_level`` regardless of how long
the trajectories are or how many restarts the analysis needs.

Usage::

    HSA_OVERRIDE_GFX_VERSION=11.0.0 python examples/llm_poc/build_pools.py \
        --samples-per-level 200 --batch-size 32 --out examples/llm_poc/pools.npz
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from humancompatible.interconnect.simulators.logic.agents.llm_agent import (
    BASE_PROMPT, NUM_LEVELS, LLMResponder, SentimentScorer, render_prompt,
)


def build(args):
    torch.set_num_threads(args.threads)
    responder = LLMResponder(model_name=args.model, device=args.device,
                             temperature=args.temperature, top_p=args.top_p,
                             max_new_tokens=args.max_new_tokens, seed=args.seed)
    scorer = SentimentScorer(device=args.device)

    pools, examples, wall = {}, {}, {}
    targets = args.only_levels if args.only_levels else list(range(NUM_LEVELS))
    if args.only_levels:
        # Merge into an existing file so a single reworded rung can be redone
        # without paying for the whole ladder again.
        prev = np.load(args.out)
        for k in range(int(prev["num_levels"])):
            pools[f"level_{k}"] = prev[f"level_{k}"]
        examples = json.loads(str(prev["examples"]))
        print(f"merging into existing {args.out}; regenerating levels {targets}")
    for level in targets:
        t0 = time.time()
        texts = []
        while len(texts) < args.samples_per_level:
            n = min(args.batch_size, args.samples_per_level - len(texts))
            texts.extend(responder.sample(level, n, args.base_prompt))
        scores = scorer.score(texts)
        wall[level] = time.time() - t0
        pools[f"level_{level}"] = scores
        examples[level] = texts[:3]
        print(f"level {level}  n={len(scores):4d}  mean={scores.mean():+.4f}  "
              f"sd={scores.std():.4f}  median={np.median(scores):+.4f}  "
              f"[{wall[level]:.1f}s, {wall[level]/len(scores):.2f}s/sample]", flush=True)
        print(f"          prompt: {render_prompt(level, args.base_prompt)!r}", flush=True)

    meta = dict(model=args.model, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, base_prompt=args.base_prompt,
                samples_per_level=args.samples_per_level, seed=args.seed,
                device=str(responder.device), total_seconds=sum(wall.values()))
    np.savez(args.out, num_levels=NUM_LEVELS, meta=json.dumps(meta),
             examples=json.dumps(examples), **pools)
    print(f"\nwrote {args.out}  ({sum(wall.values()):.1f}s total, "
          f"{len(targets) * args.samples_per_level} generations)")

    means = np.array([pools[f"level_{k}"].mean() for k in range(NUM_LEVELS)])
    monotone = bool(np.all(np.diff(means) <= 1e-9))
    print(f"per-level means: {np.round(means, 4).tolist()}")
    print(f"monotone non-increasing: {monotone}   "
          f"control authority (level 0 - level {NUM_LEVELS-1}): {means[0] - means[-1]:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--samples-per-level", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-prompt", default=BASE_PROMPT)
    ap.add_argument("--only-levels", type=int, nargs="+", default=None,
                    help="regenerate only these rungs, merging into --out")
    ap.add_argument("--out", default="examples/llm_poc/pools.npz")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
