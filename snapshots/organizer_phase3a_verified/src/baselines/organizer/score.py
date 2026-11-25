"""Post-inference ETs-only diagnostic using unchanged frozen evaluator."""
import json,sys
import hashlib
from pathlib import Path
# Python 3.10 lacks hashlib.file_digest, used only for SHA256 by the frozen adapter.
# This streaming compatibility shim changes no evaluator source or metric behavior.
if not hasattr(hashlib,'file_digest'):
    def file_digest(fileobj,name):
        h=hashlib.new(name)
        for chunk in iter(lambda:fileobj.read(1024*1024),b''):h.update(chunk)
        return h
    hashlib.file_digest=file_digest
root=Path(sys.argv[1]);work=Path(sys.argv[2]);sys.path.insert(0,str(root/'scripts'))
from evaluate_local_reference import score_strict
from contracts import write,sha
asset=json.loads((root/'configs/baselines/organizer_v1/assets.json').read_text())
gt=root/'configs/baselines/organizer_v1/scoring/ETs.csv';thresholds=root/'configs/baselines/organizer_v1/scoring/thresholds.csv'
for p in [gt,thresholds]:
    if sha(p)!=asset['scoring_hashes'][p.name]:raise ValueError('Scoring-only input hash mismatch')
result={variant:score_strict(gt,work/'predictions.csv',thresholds,variant) for variant in ['participant','prose_sensitivity']}
write(work/'I/metrics.json',{'purpose':'ETs dev_b engineering smoke diagnostic, not full-fold validation or leaderboard','metrics':result})
