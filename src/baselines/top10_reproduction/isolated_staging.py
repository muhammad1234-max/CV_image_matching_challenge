"""Administrative preparation only. The destination is a fresh export directory."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / 'configs/baselines/top10_reproduction_v1'
SEEDS = (20261111, 20261112)
PLAN_HASH = '5951162b32f22537e547b6c2d42f1df9db48666bf92d1d2c24f7a99ff054ffbf'
MODULES = set('colmap config core data data_schema distributed extractor pipeline preprocess storage workspace kernel'.split())
PACKAGES = set('clusterings features global_descriptors localizers matchers models pipelines postprocesses preprocesses retrievers shortlists utils'.split())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ordered(images, seed):
    if seed not in SEEDS:
        raise ValueError('Unknown diagnostic condition')
    # Equal bytes are interchangeable. Only the administrator uses the identity tie-break.
    canonical = sorted(images, key=lambda name: (images[name], name))
    def rank(index):
        return hashlib.sha256(f'ns64-order-v1/{seed}/{index}'.encode()).digest()
    return [canonical[i] for i in sorted(range(len(canonical)), key=rank)]


def allowed(name, config):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or any(c in name for c in ('\\', ':')):
        return False
    return name in (config, 'conf/models.yaml') or (path.suffix == '.py' and (
        (len(path.parts) == 1 and path.stem in MODULES) or path.parts[0] in PACKAGES))


def copy_checked(src, dst, expected):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if digest(dst) != expected:
        raise ValueError('Content verification failed')


def prepare(images, receipt, dest, seed):
    import ets_worker
    if digest(CFG/'pilot.json') != PLAN_HASH:
        raise ValueError('Pilot plan changed')
    plan = json.loads((CFG/'pilot.json').read_text())
    if digest(CFG/'assets_phase3c1.json') != '251d358ed39e41cd8a9daeb666dbf407fa0bbab27ee63a568df9773bbf15fecb':
        raise ValueError('Asset manifest changed')
    assets = json.loads((CFG/'assets_phase3c1.json').read_text())
    paths = ets_worker.approved_paths(receipt, assets)
    names = ordered(plan['images'], seed)
    if images.name != 'ETs' or images.parent.name != 'train' or len(names) != 22 or set(names) != {p.name for p in images.iterdir()}:
        raise ValueError('Expected exact train/ETs inventory')
    dest.mkdir(parents=True, exist_ok=False)
    package = dest/'package'
    mapping = {}
    for i, name in enumerate(names):
        opaque = f'image_{i:04d}.png'
        copy_checked(images/name, package/'test/ETs'/opaque, plan['images'][name])
        mapping[opaque] = name
    bundle = CFG/'ns64_pilot_source.zip.bin'
    if digest(bundle) != plan['source_zip_sha256']:
        raise ValueError('Published source changed')
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if allowed(name, plan['config']):
                target = package/'source'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    if digest(package/'source'/plan['config']) != plan['config_sha256']:
        raise ValueError('Configuration mismatch')
    aliases = {}
    for key, asset in assets['assets'].items():
        alias = asset['alias']
        directory = 'dinov2' if key == 'dinov2' else 'checkpoints'
        target = Path('models')/directory/asset['filename']
        src = paths[alias]/asset['filename'] if key == 'dinov2' else paths[alias]
        copy_checked(src, package/target, asset['sha256'])
        for side, sha in asset.get('sidecars', {}).items():
            copy_checked(src.parent/side, (package/target).parent/side, sha)
        aliases[alias] = (Path('/input')/(target.parent if key == 'dinov2' else target)).as_posix()
    (package/'aliases.json').write_text(json.dumps(aliases))
    shutil.copyfile(Path(__file__).with_name('isolated_worker.py'), package/'worker.py')
    inventory = {p.relative_to(package).as_posix():digest(p) for p in package.rglob('*') if p.is_file()}
    admin = dict(mapping=mapping, seed=seed, benchmark=plan['benchmark'], source_hash=digest(bundle),
                 config_hash=plan['config_sha256'], checkpoints={k:v['sha256'] for k,v in assets['assets'].items()},
                 permutation_hash=hashlib.sha256('\n'.join(plan['images'][n] for n in names).encode()).hexdigest(),
                 adapter_hash=inventory['worker.py'], dependency_receipt_hash=digest(receipt), inventory=inventory)
    (dest/'private.json').write_text(json.dumps(admin, indent=2))
    return admin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('images','receipt','dest'):
        parser.add_argument(name,type=Path)
    parser.add_argument('seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    prepare(args.images,args.receipt,args.dest,args.seed)
