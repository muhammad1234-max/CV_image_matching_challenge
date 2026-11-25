# Dioscuri/Baalshamin Kaggle compatibility review

The Kaggle preflight returned PLATFORM_DIFFERENCES. Inference did not start. The saved Phase 5A.3 Colab journal contains only the created event. The capture session-budget gate fired before a started event.

- GPU inventory shows two Tesla T4 devices with 15,360 MiB each and driver 580.178.04. The approved worker expects a single inventory row with driver 580.82.07.
- The existing resource guard passed at idle. The cgroup had a 30 GiB limit and about 0.79 GiB charged. Scratch disk had about 19.5 GiB free. These are preflight measurements.
- Base Python is 3.13.16 with PyTorch 2.11.0+cu130. The approved environment uses Python 3.11.11 and PyTorch 2.5.1+cu121. The base probe did not find pycolmap, asmk, curope or mast3r package metadata. Existing Linux CPython 3.11 wheels for pycolmap, asmk and curope are available for compatibility verification.
- The probe did not establish an empty /kaggle/input directory. It did not record whether the directory was absent or populated. The next isolation check should distinguish those cases and record relevant mounts before staging approved files.

Review a Kaggle-specific operational adapter. Require a pinned environment and verify native imports, a basic CUDA operation, the selected T4 and resource guard, clean filesystem boundaries, and checkpoint and journal export/restore. Verify the existing created journal before transitioning custody to any revised runner. Perform these checks before capture. Preserve the frozen 138-image Dioscuri/Baalshamin membership, candidate, comparator, evaluator, limits and one-trial policy.

Evidence is phase5a4_kaggle_preflight.json. Its SHA256 is ee81543ee43ae74356eeeb99afbcd4dd1763bb07c2d79dc463b4bc583fe2b432.
