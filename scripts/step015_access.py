"""Single safe CLI access check, with secret-free diagnostics."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
evidence = root / 'docs/evidence'
report = {'command': 'python -m kaggle competitions files image-matching-challenge-2025-ongoing --page-size 200 --format json'}
config = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home()/'.kaggle')))
try:
    data = json.loads((config/'kaggle.json').read_text())
    report['legacy_username_key_fields_present'] = all(bool(data.get(k)) for k in ('username','key'))
    del data
except Exception as exc:
    report['credential_inspection_error_type'] = type(exc).__name__
report['token_file_present'] = (config/'access_token').is_file()
report['credential_env_presence'] = {k: bool(os.environ.get(k)) for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN')}
report['disk_bytes'] = {d: shutil.disk_usage(d)._asdict() for d in ('C:\\','D:\\')}
report['temp_drive'] = Path(tempfile.gettempdir()).drive
report['cache_env_drive_only'] = {k: Path(os.environ[k]).drive if os.environ.get(k) else 'unset' for k in ('PIP_CACHE_DIR','HF_HOME','TORCH_HOME','UV_CACHE_DIR','TEMP','TMP')}
for args in [['--version'],['auth','login','--help'],['competitions','files','--help'],['competitions','submit','--help']]:
    p=subprocess.run([sys.executable,'-m','kaggle',*args],capture_output=True,text=True,timeout=20)
    name='_'.join(x.strip('-') for x in args)
    (evidence/f'step015_cli_{name}.txt').write_text(p.stdout,encoding='utf-8')
try:
    p=subprocess.run([sys.executable,'-m','kaggle','competitions','files','image-matching-challenge-2025-ongoing','--page-size','200','--format','json'],capture_output=True,text=True,timeout=45)
    report['exit_code']=p.returncode
    combined=p.stdout+'\n'+p.stderr
    status=re.search(r'\b(401|403|429)\b',combined)
    report['http_status']=int(status.group(1)) if status else None
    report['status']='SUCCESS' if p.returncode==0 else 'NEEDS USER ACTION' if report['http_status']==401 else 'FAILED'
    if p.returncode==0:
        report['listing_output']=json.loads(p.stdout)
    else:
        report['error']='Raw CLI output suppressed; no credential values or sensitive paths retained.'
except subprocess.TimeoutExpired:
    report['status']='FAILED'
    report['error']='CLI timed out after 45 seconds; no retry.'
(evidence/'step015_access.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
