"""Sample the LLM at every instruction rung once and save the sentiment scores ("pools").

The loop is a Markov chain on the rung k, and responses are i.i.d. given k, so the pools are all that
ladder_results.ipynb needs: the LLM cost is rungs x samples, however long the simulated runs.

    python build_pools.py                                          # signed ladder, rungs 0..6
    python build_pools.py --ladder repeat --out pools_repeat.npz   # k copies of "Be more neutral."
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath("../.."))
import numpy as np
import torch
from humancompatible.interconnect.simulators.logic.agents.llm_agent import (
    BASE_PROMPT, LADDERS, NUM_LEVELS, LLMResponder, SentimentScorer)

ap = argparse.ArgumentParser()
ap.add_argument("--ladder", choices=sorted(LADDERS), default="signed")
ap.add_argument("--samples", type=int, default=200)
ap.add_argument("--out", default="pools.npz")
args = ap.parse_args()

# repeat ladder has no last rung: 0..9 for the near field, 20 and 50 to check the law has stopped changing
LEVELS = list(range(NUM_LEVELS)) if args.ladder == "signed" else list(range(10)) + [20, 50]
META = dict(model="Qwen/Qwen2.5-0.5B-Instruct", temperature=1.0, top_p=0.95, max_new_tokens=28, seed=0,
            base_prompt=BASE_PROMPT, samples_per_level=args.samples, ladder=args.ladder)
BATCH = 32

torch.set_num_threads(16)
llm = LLMResponder(model_name=META["model"], temperature=META["temperature"], top_p=META["top_p"],
                   max_new_tokens=META["max_new_tokens"], seed=META["seed"], render=LADDERS[args.ladder])
scorer = SentimentScorer()   # s = P(positive) - P(negative) in [-1, 1]

pools, examples = {}, {}
for i, k in enumerate(LEVELS):
    texts = [t for b in range(0, args.samples, BATCH) for t in llm.sample(k, min(BATCH, args.samples - b))]
    pools[f"level_{i}"] = s = scorer.score(texts)
    examples[str(k)] = texts[:3]
    print(f"rung {k:3d}: mean s = {s.mean():+.3f}, sd = {s.std():.3f}", flush=True)

np.savez(args.out, num_levels=len(LEVELS), levels=np.array(LEVELS), ladder=args.ladder,
         meta=json.dumps(META), examples=json.dumps(examples), **pools)
print(f"wrote {args.out}")
