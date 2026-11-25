# Environment, assets and reproduction boundaries

Exact original algorithm wheels acquired and hash-pinned:

| Distribution | Version | Required wheel ABI |
|---|---|---|
| kornia | 0.7.2 | pure Python |
| kornia-rs | 0.1.2 | cp310 Linux manylinux2014 x86_64 |
| lightglue | 0.0 | pure Python, exact wheel hash is essential |
| pycolmap | 0.6.1 | cp310 Linux manylinux2014 x86_64 |

The source package bundle also has kornia-moons 0.2.9 and rerun-sdk 0.15.0a2; they are not used by the imported algorithm functions and are not installed by the smoke harness. Source Python metadata is 3.10.12. PyTorch, CUDA, Transformers, NumPy, torchvision, h5py and Pillow were not exactly pinned in the notebook source. They remain runtime facts to capture, not invented exact dependencies. Imports, versions, CUDA availability and pycolmap option defaults are recorded in Stage A. The harness requires Linux CPython 3.10 and conservatively requires NumPy 1.x for this historical compiled stack; it does not silently downgrade an existing environment.

The old wheels cannot load under Python 3.13. Previous returned CPU audits used Python 3.13, but the current GPU runtime has not been executed/observed. Compatibility with a current Kaggle GPU image is therefore **unverified and likely blocked unless a compatible environment is available**. The original container digest is in source_audit.json; no claim is made that Kaggle allows selecting it today. See official platform source https://github.com/Kaggle/docker-python and pycolmap 0.6.1 release metadata https://pypi.org/project/pycolmap/0.6.1/ . The old PyPI wheels are not CUDA-enabled COLMAP builds; learned features/matching use PyTorch CUDA while reconstruction can be CPU-bound. We preserve the original match_exhaustive call rather than substituting a different geometric verifier.

Required model mounts, from original notebook metadata:

- `metaresearch/dinov2/PyTorch/base/1`: local config.json, preprocessor_config.json and .bin/.safetensors weights.
- `oldufo/aliked/PyTorch/aliked-n16/1`: aliked-n16.pth.
- `oldufo/lightglue/PyTorch/aliked/1`: aliked_lightglue.pth.

No model weights were downloaded locally. On first mounted preflight, every checkpoint/config is hashed, written to checkpoint_hashes.json before inference and copied into the expected local Torch cache. On resume, changed bytes fail. These are observed content pins, not independently authenticated publisher hashes. The first model run cannot establish exact reproduction until mounted checkpoint identity and the environment are reviewed. Runtime downloads are disabled through HuggingFace offline flags and a failing Torch downloader. Internet should stay OFF.

Resource settings: one GPU (cuda:0), default 12 GiB PyTorch allocator budget, at most 1200 seconds including setup. This is not a guarantee of total process/device memory: CUDA contexts and native allocators lie outside the PyTorch cap. T4 is a reasonable first platform because the expanded source metadata used T4; actual sufficiency is untested. CPU reconstruction may dominate time. A timeout preserves prior completed stages and fails the active one; OOM preserves worker error, peak allocated/reserved bytes and active configuration. CPU peak RSS and GPU peaks are per worker; driver/GPU metadata are captured at preflight. Exact CUDA/torch binary compatibility must pass real imports/inference.

Explicit adaptations (not novel model changes): ongoing competition input rather than original closed competition; one intact ETs dataset; shuffled opaque input names and lossless RGB re-encoding with sanitized metadata; staged processes, logging, caching, caps and fail-fast behavior; fixed random seeds; local frozen metric replacing organizer's unpinned metric; SHA256-only streaming compatibility shim for Python 3.10 because the frozen adapter calls Python 3.11's hashlib.file_digest. No frozen evaluator file is modified. NumPy/GPU environment differs from the CPU evaluator verification environment and must be recorded before baseline registration. These boundaries mean **architecture-faithful adapted smoke harness, not demonstrated exact end-to-end reproduction**.

No GPU packages were installed on Windows, no paid resources were provisioned and no model was run. Do not implement a pycolmap upgrade as an implicit workaround. Return a blocked preflight report for review if a compatible Python 3.10 runtime is unavailable.
