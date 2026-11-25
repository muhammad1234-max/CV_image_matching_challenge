# Environment and resource controls

Use the organizer fork's **Pin to original environment**, Python **3.10.12**, **Tesla T4**, Internet **OFF**. One GPU is used even if the UI offers T4 ×2. Python 3.13 is deliberately blocked. Do not upgrade or substitute packages.

Locked core: PyTorch 2.5.1+cu121, CUDA build 12.1, NumPy 1.26.4, pycolmap 0.6.1, Kornia 0.7.2, kornia-rs 0.1.2, LightGlue 0.0, Transformers 4.47.0, torchvision 0.20.1+cu121, h5py 3.12.1, Pillow 11.0.0, OpenCV 4.10.0.84. Original wheel SHA256 values are checked before offline installation. Full package lists and `nvidia-smi` output are recorded; OS/driver updates outside the locked core remain visible environmental deviations, not guaranteed bitwise equivalence.

Mount ongoing competition data, original `oldufo/imc2024-packages-lightglue-rerun-kornia`, and these model versions:

- `metaresearch/dinov2/PyTorch/base/1`
- `oldufo/aliked/PyTorch/aliked-n16/1`
- `oldufo/lightglue/PyTorch/aliked/1`

Both modern `/kaggle/input/models/...` / `/kaggle/input/competitions/...` and historical input paths are supported. Ambiguous mounts stop. DINO configuration, preprocessing, weights and accompanying files plus both local model weights must match the successful smoke checkpoint inventory exactly.

The frozen per-process PyTorch limit is 12 GiB. Worker measurements record allocated/reserved CUDA peaks and CPU `ru_maxrss`. Parent sampling records process-tree RSS and total device memory, which can include other processes and miss transient peaks. No missing measurement is inferred from ETs. Available disk/RAM are recorded; a conservative feature/database disk allowance is checked per dataset. Full-development resource use is currently unknown.

Set `SESSION_REMAINING_SECONDS` to actual remaining platform time, at most 21,600; it must exceed the 600-second archive reserve. The six-hour cap is our operational limit, not a claim about Kaggle's current account/session quota. A 7,200-second SfM wall-clock cap does not alter SfM parameters. Timeouts and OOMs produce incomplete results, never fabricated successful poses. Completed stage hashes permit an identical retry. Archive creation is best-effort within the reserve; avoid launching near a platform shutdown.

Phase 3A diagnostic only: 253.80 seconds wall time, 1.58 GiB allocated and 2.69 GiB reserved on ETs. None of these constitutes measured full-development compute.
