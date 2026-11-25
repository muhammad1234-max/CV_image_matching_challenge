"""Prepare only opaque image bytes on the trusted project computer."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/baselines/top10_reproduction'))
from isolated_staging import CFG, PLAN_HASH, SEEDS, digest, ordered


def export(source, destination):
    plan = CFG/'pilot.json'
    if digest(plan) != PLAN_HASH:
        raise ValueError('Pilot inventory changed')
    images = json.loads(plan.read_text())['images']
    private = {}
    with zipfile.ZipFile(source) as z:
        if sorted(z.namelist()) != sorted(images):
            raise ValueError('Expected exactly the 22 frozen ETs images')
        data = {name:z.read(name) for name in images}
    if any(hashlib.sha256(data[n]).hexdigest()!=images[n] for n in images):
        raise ValueError('Image hash mismatch')
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(destination/'opaque_inputs.zip','x') as z:
        for seed in SEEDS:
            mapping = {}
            for i,name in enumerate(ordered(images,seed)):
                opaque = f'image_{i:04d}.png'
                z.writestr(f'{seed}/{opaque}',data[name])
                mapping[opaque] = name
            private[str(seed)] = mapping
    (destination/'private.json').write_text(json.dumps(private,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    export(args.source,args.destination)
