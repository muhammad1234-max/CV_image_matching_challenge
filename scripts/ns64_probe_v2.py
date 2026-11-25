"""Phase 3C.4 administrative preparation and non-inference container launcher."""
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/baselines/top10_reproduction'))
import isolated_staging as staging
import verify_ns64_isolation as previous

CONFIG = ROOT/'configs/baselines/ns64_isolated_v1/experiment.json'
LOCK = ROOT/'configs/baselines/ns64_probe_v2/source.json'
PROBE = Path(__file__).with_name('ns64_container_readiness.py')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_entries(entries):
    seen=set()
    for item in entries:
        name=item.orig_filename.rstrip('/')
        path=PurePosixPath(name)
        kind=stat.S_IFMT(item.external_attr >> 16)
        if (path.as_posix()!=name or path.is_absolute() or '..' in path.parts or
            any(c in name for c in ('\\',':')) or name.casefold() in seen or
            kind not in (0,stat.S_IFDIR,stat.S_IFREG)):
            raise ValueError('Unsafe or duplicate source archive member')
        seen.add(name.casefold())


def verify_package(directory):
    package=previous.checked_package(directory)
    admin=json.loads((directory/'private.json').read_text())
    lock=json.loads(LOCK.read_text())
    config=json.loads(CONFIG.read_text())
    seed=admin['seed']
    expected=config['conditions'][f'ns64_isolated_v1_ETs_{seed}']['permutation_sha256']
    inventory=admin['inventory']
    source={n[7:]:v for n,v in inventory.items() if n.startswith('source/')}
    names=[f'test/ETs/image_{i:04d}.png' for i in range(22)]
    hashes=[inventory[n] for n in names]
    permutation=hashlib.sha256('\n'.join(hashes).encode()).hexdigest()
    assets=json.loads((staging.CFG/'assets_phase3c1.json').read_text())['assets']
    expected_assets={}
    aliases={}
    for key,asset in assets.items():
        folder='models/dinov2' if key=='dinov2' else 'models/checkpoints'
        weight=folder+'/'+asset['filename']
        expected_assets[weight]=config['assets'][key]
        expected_assets.update({folder+'/'+n:h for n,h in asset.get('sidecars',{}).items()})
        aliases[asset['alias']]='/input/'+(folder if key=='dinov2' else weight)
    entries=set(names)|{'worker.py','aliases.json','probe_v2.py'}|set(expected_assets)|{'source/'+n for n in lock}
    if (source!=lock or set(inventory)!=entries or permutation!=expected or admin['permutation_hash']!=expected or
        any(inventory[n]!=h for n,h in expected_assets.items()) or json.loads((package/'aliases.json').read_text())!=aliases or
        inventory['worker.py']!=config['source_files']['src/baselines/top10_reproduction/isolated_worker.py'] or
        inventory['probe_v2.py']!=staging.digest(PROBE)):
        raise ValueError('Package differs from reviewed condition')
    return package, dict(seed=seed,permutation_hash=permutation,package_hash=fingerprint(inventory),
                         source_inventory_hash=fingerprint(source), checkpoint_hashes=config['assets'],
                         worker_hash=inventory['worker.py'],probe_hash=inventory['probe_v2.py'])


def prepare(args):
    with zipfile.ZipFile(staging.CFG/'ns64_pilot_source.zip.bin') as archive:
        validate_entries(archive.infolist())
    # This fresh administrative workspace is never mounted during preparation.
    admin=staging.prepare(args.images,args.receipt,args.directory,args.seed)
    package=args.directory/'package'
    unwanted='source/utils/example.py'
    (package/unwanted).unlink()
    del admin['inventory'][unwanted]
    (package/'probe_v2.py').write_bytes(PROBE.read_bytes())
    admin['inventory']['probe_v2.py']=staging.digest(PROBE)
    (args.directory/'private.json').write_text(json.dumps(admin))
    verify_package(args.directory)


def run(args):
    package,record=verify_package(args.directory)
    output=args.output.resolve()
    if output.is_relative_to(args.directory.resolve()):
        raise ValueError('Use separate output directory')
    output.mkdir(parents=True,exist_ok=False)
    output.chmod(0o777)
    runtime=json.loads(subprocess.check_output(['docker','image','inspect',args.image]))[0]
    if runtime['Id']!=args.image or runtime['Config'].get('Volumes'):
        raise ValueError('Unexpected image identity or declared mount')
    cmd=previous.command(args.image,package,output)
    cmd[-1]='/input/probe_v2.py'
    record.update(status='BLOCKED',runtime_image_id=args.image,host_os=os.uname().sysname+' '+os.uname().release,
                  docker_server=subprocess.check_output(['docker','version','--format','{{.Server.Version}}'],text=True).strip(),
                  controls=cmd[4:cmd.index('--mount')])
    try:
        with tempfile.TemporaryDirectory(prefix='ns64-admin-') as tmp:
            marker=Path(tmp)/'canary'
            marker.write_text('public probe marker')
            record['host_canary_verified']=marker.read_text()=='public probe marker'
            proc=subprocess.run(cmd+[str(marker),record['package_hash']],capture_output=True,text=True,timeout=300)
            # Keep diagnostics local. Export only validated structured status and exit code.
            record['exit_code']=proc.returncode
            record['stdout_bytes']=len(proc.stdout.encode())
            record['stderr_bytes']=len(proc.stderr.encode())
            if proc.returncode:
                raise RuntimeError('Probe failed; inspect local container status')
            result=json.loads((output/'readiness.json').read_text())
            if result['status']!='PASS' or result['package_hash']!=record['package_hash']:
                raise ValueError('Invalid readiness receipt')
            record.update(status='PASS',readiness_hash=staging.digest(output/'readiness.json'),
                          boundary_hash=staging.digest(output/'boundary.json'))
    finally:
        (output/'host.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ('images','receipt','directory'):
        prep.add_argument(name,type=Path)
    prep.add_argument('seed',type=int,choices=staging.SEEDS)
    probe=sub.add_parser('probe')
    probe.add_argument('directory',type=Path)
    probe.add_argument('image')
    probe.add_argument('output',type=Path)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args)

