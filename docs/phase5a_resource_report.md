# Resource preflight

The execution-integrity gate stopped the experiment before allocating GPU time. The local Windows 3.13.7 workstation exposed 8,465,264,640 bytes of RAM, with 2,986,921,984 bytes available at sampling and 134,229,954,560 bytes free on the project volume. nvidia-smi and Linux /proc/meminfo were unavailable. The resource receipt records the UTC measurement time. These values describe the administrator workstation only.

A clean remote GPU, free VRAM, cgroup limit and current charge, host availability, disk, swap, memory pressure, OOM history, exact CUDA/Python and remaining platform lifetime were unmeasured. The signed input archive is 313,857,668 bytes. The worker's initial scratch estimate from the uncompressed image lengths is 1,255,373,176 bytes. Actual peak scratch also depends on the numerical parent, scratch copies, mapper outputs, opaque image staging, ZIP export and the separately installed runtime. The initial estimate alone cannot establish sufficient capacity.

## Conditional Colab runbook

Keep the inference cells on hold pending review of the checkpoint wrapper. For an authorized successor, use the notebook and protocol hashes specified in its release. Select a fresh free T4 session with Drive unmounted. Supply only imc_ns64_clean_runtime.zip during setup and imc2024_dioscuri_baalshamin_opaque.zip during input staging. Leave the dataset at its frozen default. Keep this administrative review ZIP and the private maps off the VM.

The following preflight-only cell uses the existing notebook's prepared kit, base, python, dataset and deadline variables after setup and staging. It reports current measurements and verifies the weight digests. Run it only in the prepared clean runtime. It launches no inference.

```python
import json, os, shutil, subprocess, time
from pathlib import Path
from src.novel_model.resource_monitor_v2 import sample, guard
from src.novel_model.geometry_capture_v1.database import file_sha

assert dataset == "imc2024_dioscuri_baalshamin"
limits = json.loads((kit/"configs/novel_model/phase4g1_resource_policy.yaml").read_bytes())
row = sample(os.getpid())
row["remaining_seconds"] = deadline - time.monotonic()
row["disk_free_bytes"] = shutil.disk_usage("/content").free
row["headroom_by_cgroup_bytes"] = [
    g["limit"] - g["current"] for g in row["groups"] if g["limit"] is not None
]
row["runtime"] = subprocess.check_output([str(python), "-c",
    "import sys,torch,pycolmap; print(sys.version,torch.__version__,torch.version.cuda,pycolmap.__version__)"], text=True).strip()
row["device_driver"] = subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.free","--format=csv,noheader"],text=True).strip()
assets = json.loads((base/"assets.json").read_bytes())
row["asset_checks"] = {}
for key, asset in assets.items():
    folder = base/"weights"/("dinov2" if key == "dinov2" else "")
    pins = {asset["filename"]: asset["sha256"], **asset.get("sidecars", {})}
    row["asset_checks"][key] = all((folder/n).is_file() and file_sha(folder/n)==h for n,h in pins.items())
print(json.dumps(row,indent=2))
guard(row,row,limits)
assert row["remaining_seconds"] >= 8400
assert row["host"]["MemAvailable"] >= 8*1024**3
assert row["disk_free_bytes"] >= 1255373176
assert all(row["asset_checks"].values())
```

The operator-declared remaining lifetime is an upper bound. Require the exact frozen dependency and driver checks and the worker’s VRAM floor of 9,412,018,176 free bytes; re-evaluate host and cgroup headroom immediately before capture. Keep the 4G.1 absolute and fractional margins, OOM-event and pressure checks active. Summed RSS can double count shared pages. Preserve PSS, USS, device memory and cgroup charge separately.

After a reviewed release passes those checks, run capture, original, trial and decision one cell at a time. Download and validate each stage ZIP and save its separately printed digest before advancing. On reset restore completed stage ZIPs in order. Preserve a failed attempt for review. The limits are 14,400 cumulative stage seconds, one trial of up to 5,400 seconds and a 1,200-second export reserve. Administrative scoring starts after final selection is sealed. Phase 5A authorizes only the 138-image group.