"""Run one shard of the Decision Index suite through FireworksJevEngine.

The kit's runner is sequential; the notebook launches several of these in parallel, each owning
the requests whose run_id hashes to its shard, then merges their results.jsonl files for scoring.
Excluded questions are skipped because they are never scored.
"""

import argparse
import hashlib
import json

from decision_index.runner import run
from decision_index.suite.io import Suite

p = argparse.ArgumentParser()
p.add_argument("--suite-dir", required=True)
p.add_argument("--shard", type=int, required=True)
p.add_argument("--n-shards", type=int, required=True)
p.add_argument("--out", required=True)
p.add_argument("--engine-options", required=True, help="JSON dict")
p.add_argument("--engine", default="jev_di_engine:FireworksJevEngine")
a = p.parse_args()

suite = Suite(a.suite_dir, "0.2")
excluded = set(suite.excluded())


def keep(e):
    if e["run_id"] in excluded or not suite.in_edition(e):
        return False
    return int(hashlib.sha256(e["run_id"].encode()).hexdigest(), 16) % a.n_shards == a.shard


print(json.dumps(run(a.engine, json.loads(a.engine_options), suite.row_paths,
                     a.out, compact=True, keep=keep, halt_on_device_error=False)))
