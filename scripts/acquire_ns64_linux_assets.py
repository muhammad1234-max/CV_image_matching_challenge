"""Acquire permitted official checkpoints on designated Linux storage, then hash and pin."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def acquire(manifest_path, destination):
    if sys.platform != 'linux':
        raise ValueError('Use the designated Linux storage volume')
    manifest=json.loads(manifest_path.read_text())
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['superpoint','retrieval','codebook','isc','mast3r']:
        asset=manifest['assets'][name]
        path=destination/asset['filename']
        tmp=path.with_suffix(path.suffix+'.partial')
        # Only accept a previously completed download with a known reference hash.
        if path.exists():
            with path.open('rb') as handle:
                actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if not asset['sha256'] or asset['sha256']!=actual:
                raise ValueError('Existing file requires independent reference verification: '+str(path))
            continue
        from shutil import disk_usage
        if disk_usage(destination).free < asset['size']:
            raise OSError('Insufficient free space for '+name)
        h=hashlib.sha256()
        with urlopen(Request(asset['url']),timeout=60) as response, tmp.open('wb') as output:
            for chunk in iter(lambda:response.read(2**20), b''):
                output.write(chunk)
                h.update(chunk)
        if tmp.stat().st_size!=asset['size'] or asset['sha256'] not in (None,h.hexdigest()):
            raise ValueError('Size or hash mismatch: '+name)
        asset['sha256']=h.hexdigest()
        evidence=destination/(name+'.authority.json')
        evidence.write_text(json.dumps({'url':asset['url'],'size':asset['size'],'sha256':asset['sha256']},indent=2))
        tmp.replace(path)
        print(name,asset['sha256'],flush=True)
    (destination/'assets_resolved.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    acquire(args.manifest,args.destination)
