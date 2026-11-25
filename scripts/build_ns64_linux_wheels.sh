#!/usr/bin/env bash
# Run inside an isolated Linux build environment with the target GPU visible.
set -euo pipefail
python -c "import sys,torch; assert sys.version_info[:2]==(3,11) and torch.__version__=='2.5.1+cu121'; assert torch.cuda.get_device_capability()==(7,5)"
export CC=gcc-11 CXX=g++-11 CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=7.5
build=$(mktemp -d)
out=$(realpath -m "${1:-ns64_wheelhouse}")
mkdir -p "$out"
cd "$build"
{ "$CC" --version; "$CUDA_HOME/bin/nvcc" --version; python -VV; python -m pip freeze; } > "$out/build-environment.txt"
git clone https://github.com/jenicek/asmk.git
git -C asmk checkout --detach 2a96d9c03a841dffdfddabc699a20512dcd09363
git clone https://github.com/naver/croco.git
git -C croco checkout --detach d7de0705845239092414480bd829228723bf20de
python -m pip install Cython==0.29.37
sha256sum asmk/cython/hamming.pyx asmk/cython/hamming.c > "$out/asmk-before.txt"
python -m cython asmk/cython/hamming.pyx
sha256sum asmk/cython/hamming.pyx asmk/cython/hamming.c > "$out/asmk-after.txt"
python -m pip wheel --no-deps --no-build-isolation -w "$out" ./asmk ./croco/models/curope
python -m pip install --no-index --no-deps --find-links="$out" asmk curope
python -c 'import asmk.hamming, curope'
(cd "$out" && sha256sum *.whl > SHA256SUMS)
