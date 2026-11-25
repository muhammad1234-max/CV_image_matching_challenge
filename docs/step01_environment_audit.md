# Step 1 — Environment audit

Audit date: **2026-11-10, Asia/Karachi**. Hardware figures are observations, not vendor specifications. Available memory and disk are point-in-time measurements.

## Workspace and execution context

`Get-ChildItem -Force` in `D:\CV_Project` returned no entries before changes. No `AGENTS.md` was found at the project or drive root. `git status --short --branch` returned `fatal: not a git repository (or any of the parent directories): .git`. No existing work was overwritten; no Git initialization, checkout, installation, training, submission, or paid provisioning occurred.

**Inference:** this is a local Windows HP laptop running sandboxed desktop commands, rather than a Kaggle/Colab runtime. Evidence: Windows executable paths, hardware manufacturer/model, and absent Kaggle/Colab/WSL/container markers in [environment.json](evidence/environment.json). CIM reports `HypervisorPresent: true`; this does not establish a cloud VM or container. Docker and WSL executables exist, but their operational state was not tested.

## Measured hardware

| Item | Observation | Diagnostic |
|---|---|---|
| Computer | HP ZBook 15 G2, Hewlett-Packard | BIOS registry and `Get-CimInstance Win32_ComputerSystem` |
| OS | Windows 10 Pro 22H2, build 19045.6466 | Python platform and Windows NT registry |
| Architecture | AMD64 / 64-bit Python | `platform.machine()`, `sys.version` |
| CPU | Intel Core i7-4810MQ @ 2.80 GHz | CPU registry and `Win32_Processor` |
| Physical/logical cores | 4 / 8 | `Win32_Processor`; `os.cpu_count()` confirms 8 logical |
| Usable physical RAM | 8,465,264,640 bytes = 7.884 GiB | Win32 `GlobalMemoryStatusEx` |
| Available RAM | 1,324,138,496 bytes = 1.233 GiB; memory load 84% | Same snapshot |
| C: total/free | 97.09 / 2.43 GiB | `shutil.disk_usage('C:\\')` |
| D: total/free | 140.82 / 132.37 GiB | `shutil.disk_usage('D:\\')` |
| NVIDIA GPU | Quadro K1100M, 2,147,483,648 bytes = 2 GiB | Registry and `Win32_VideoController` |
| NVIDIA driver | Windows version 22.21.13.8573 | Same; no inferred CUDA compatibility |
| Integrated GPU | Intel HD Graphics 4600, driver 20.19.15.5126 | Same; reported 1 GiB AdapterRAM is not verified dedicated VRAM |
| CUDA | Torch build `None`; availability `False` | `torch.version.cuda`, `torch.cuda.is_available()` |
| CUDA toolkit/runtime | `nvcc`, `nvidia-smi`, `CUDA_PATH` unavailable | PATH/env checks; standalone runtime installation unknown |

Commands and outputs: [initial failures](evidence/initial_diagnostics.txt), [structured audit](evidence/environment.json), [confirmed checks](evidence/confirmed_diagnostics.txt). CIM and volume commands initially failed with Access denied inside the sandbox; read-only CIM checks outside it succeeded. `nvidia-smi` was also absent at the two conventional locations checked. Its absence does not imply absence of an NVIDIA GPU.

## Software inventory

| Component | Result |
|---|---|
| Python | 3.13.7, `C:\Python313\python.exe`; `py -0p` lists only 3.13 |
| pip | 26.1.2; executable and user-site distribution paths differ |
| Git | 2.47.1.windows.1 |
| uv | PowerShell finds a WinGet link, but execution fails: no application associated; Python `which` returns null |
| conda / mamba | Not found on PATH |
| torch / torchvision | 2.10.0+cpu / 0.25.0+cpu; imports confirmed outside sandbox |
| OpenCV | Imported `cv2` 4.13.0; distributions include opencv-contrib-python 4.13.0.92 and opencv-python 4.10.0.84 |
| Kornia / pycolmap / faiss | Imports fail with ModuleNotFoundError; distribution metadata absent |
| COLMAP executable | Not on PATH; other installations not exhaustively searched |
| transformers | 5.17.0; import confirmed |
| kaggle | Distribution 2.2.4; API import and credential loading succeed; CLI not on PATH |
| NumPy | 2.1.2 |

The overlapping OpenCV distributions and global Python environment are reproduction risks. Initial 8-second import timeouts were resolved by a longer, read-only external check; they must not be interpreted as missing packages. The audit script's first JSON serialization failed on binary registry data, then was fixed and rerun successfully. These failures are retained in the evidence notes.

## Reproduction plan — not executed

Use an isolated Linux/Python 3.11 environment for the first-place candidate. Its repository documents Python 3.11 and missing locally built ASMK/curope wheels; the dependency manifest pins torch 2.5.1, torchvision 0.20.1, transformers 4.47.0 and NumPy <2. This differs materially from this machine. Sources: [README](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025), [manifest](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/pyproject.toml).

After supervisor review: select compute, pin a source commit, audit licenses and build instructions, record OS/Python/compiler/driver/CUDA and wheel hashes, create an environment lock, and hash model weights and data. Preserve notebook version and offline inputs. Do not treat today's mutable `main` snapshots as a reproducible full checkout. Diagnostic scripts use the standard library except `audit_kaggle.py`, which uses the already-installed Kaggle package. No requirements file claiming a working model environment was fabricated.

## Step 1.5 review addendum

Original findings above are retained as the Step 1 record. The later [supervisor review](step015_supervisor_review.md) qualifies CLI availability/authentication, evaluator validation and current rule uncertainty. The [execution plan](step015_execution_environment.md) supersedes the initial baseline order with starter-first integration, followed by a verified top-10 target. Current readiness: [BLOCKED](step015_step2_readiness.md).
