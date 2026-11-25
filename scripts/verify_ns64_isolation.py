"""Admin-side CPU probe launcher. Requires a reviewed clean Linux runtime image."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def checked_package(directory):
    package = directory/'package'
    record = json.loads((directory/'private.json').read_text())
    files = list(package.rglob('*'))
    if package.is_symlink() or any(p.is_symlink() for p in files):
        raise ValueError('Unexpected package symlink')
    actual = {}
    for path in files:
        if path.is_file():
            with path.open('rb') as handle:
                actual[path.relative_to(package).as_posix()] = hashlib.file_digest(handle,'sha256').hexdigest()
    if not actual or actual != record['inventory']:
        raise ValueError('Invalid package inventory')
    return package.resolve()


def command(image, package, output):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image):
        raise ValueError('Use an inspected local image ID')
    if any(',' in str(p) for p in (package,output)):
        raise ValueError('Invalid mount path')
    return ['docker','run','--rm','--pull=never','--network=none','--read-only','--user=65534:65534',
            '--cap-drop=ALL','--security-opt=no-new-privileges','--ipc=private',
            '--tmpfs','/tmp:rw,nosuid,nodev,mode=1777','--workdir=/output','--env','HOME=/tmp',
            '--env','MPLBACKEND=Agg','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1',
            '--mount',f'type=bind,src={package},dst=/input,readonly',
            '--mount',f'type=bind,src={output},dst=/output','--gpus=device=0',
            '--entrypoint=python',image,'-I','/input/worker.py']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('image')
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    package = checked_package(args.directory)
    output = args.output.resolve()
    if output.is_relative_to(args.directory.resolve()):
        raise ValueError('Use separate empty output')
    output.mkdir(parents=True,exist_ok=False)
    output.chmod(0o777)
    info = json.loads(subprocess.check_output(['docker','image','inspect',args.image]))[0]
    if info['Id'] != args.image or info['Config'].get('Volumes'):
        raise ValueError('Unexpected image or implicit mounts')
    subprocess.run(command(args.image,package,output),check=True,timeout=120)
    evidence = json.loads((output/'boundary.json').read_text())
    if evidence['status'] != 'PASS':
        raise RuntimeError('Worker isolation blocked')
    evidence['runtime_image_id'] = args.image
    (output/'boundary.json').write_text(json.dumps(evidence,indent=2))
    print('Probe passed',args.image)

