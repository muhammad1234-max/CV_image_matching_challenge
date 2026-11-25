"""Make a data-free Docker build context from the historical dependency bundle."""
import argparse
import email
import hashlib
import io
import re
import zipfile
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('migration',type=Path)
parser.add_argument('context',type=Path)
parser.add_argument('ubuntu_base',help='Reviewed Ubuntu 24.04 amd64 image reference pinned by digest')
args=parser.parse_args()
if not re.fullmatch(r'(?:docker.io/library/)?ubuntu@sha256:[a-f0-9]{64}',args.ubuntu_base):
    raise ValueError('A digest-pinned Ubuntu base is required')
with args.migration.open('rb') as f:
    if hashlib.file_digest(f,'sha256').hexdigest()!='781dffd54912ca6955f4d83b652afef55b9f9293c06510c18a7f7291573e4698':
        raise ValueError('Unexpected migration ZIP')
args.context.mkdir(parents=True,exist_ok=False)
wheel_dir=args.context/'wheels'
wheel_dir.mkdir()
names={}
def add(data,name):
    target=wheel_dir/Path(name).name
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
    key=metadata['Name'].lower().replace('_','-')
    if key in names:
        raise ValueError('Duplicate wheel')
    names[key]='/build/wheels/'+target.name
    target.write_bytes(data)
with zipfile.ZipFile(args.migration) as z:
    for name in z.namelist():
        if name.startswith('kaggle_upload/ns64_linux_cp311/') and name.endswith('.whl'):
            add(z.read(name),name)
    with zipfile.ZipFile(io.BytesIO(z.read('configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin'))) as source:
        for name in source.namelist():
            if name.startswith('bundle/oss/') and name.endswith('.whl'):
                add(source.read(name),name)
    requirements=[]
    for line in z.read('colab/kaggle_environment.txt').decode().splitlines():
        if ' @ ' in line:
            line=names[line.split(' @ ')[0].lower().replace('_','-')]
        elif not re.fullmatch(r'[\w.-]+==[\w.+-]+',line):
            raise ValueError('Unexpected requirement')
        requirements.append(line)
(args.context/'requirements.txt').write_text('\n'.join(requirements)+'\n')
(args.context/'Dockerfile').write_text(f"""FROM {args.ubuntu_base}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv ca-certificates libgl1 libglib2.0-0 libgomp1 libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/bootstrap && /opt/bootstrap/bin/pip install uv==0.13.8
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN /opt/bootstrap/bin/uv python install 3.11.11 && /opt/bootstrap/bin/uv venv --seed --python 3.11.11 /opt/ns64
ENV PATH=/opt/ns64/bin:$PATH
COPY wheels /build/wheels
COPY requirements.txt /build/requirements.txt
RUN pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 -r /build/requirements.txt
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
""")
