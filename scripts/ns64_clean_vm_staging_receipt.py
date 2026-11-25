"""Record post-staging checks in the existing clean Colab VM."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile

content = Path("/content")
receipt = {"status": "BLOCKED", "scope": "clean VM post-staging checks"}
expected = "0d8e17647d39dfcb731ec4de158b9079506a03f71620ecefb392351f5a9d3831"
try:
    previous = content / "imc_ns64_clean_vm_precheck.json"
    pre = json.loads(previous.read_bytes())
    receipt["precheck_sha256"] = hashlib.sha256(previous.read_bytes()).hexdigest()
    if pre["status"] != "CLEAN_VM_PRECHECK_PASS" or pre["probe_sha256"] != "7186cf6f7a670191f5aca873c3f89be43b972ed75b4cc1edf98634d741c48551":
        raise ValueError("Expected verified precheck")
    inputs = Path("/kaggle/input")
    if inputs.exists() and (inputs.resolve() != inputs or any(inputs.iterdir())):
        raise ValueError("Unexpected Kaggle input")
    paths = set(pre["administrative_paths"]) - {str(inputs)}
    paths.update(["/content/ns64/data/train/ETs", "/content/ns64/project",
                  "/content/ns64/project/configs/baselines/top10_reproduction_v1/pilot.json",
                  "/proc/1/root/content/drive", "/var/run/docker.sock"])
    checks = receipt["path_checks"] = {}
    for name in sorted(paths):
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK)
        except (FileNotFoundError, PermissionError, NotADirectoryError) as error:
            checks[name] = type(error).__name__
        else:
            os.close(fd)
            checks[name] = "READABLE"
            raise ValueError("Administrative path accessible")
    archives = sorted(content.glob("opaque_inputs*.zip"))
    if not archives:
        raise FileNotFoundError("Staged ZIP missing")
    for archive in archives:
        if archive.is_symlink() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError("Unexpected staged archive")
    receipt["archive_sha256"] = expected
    orders = receipt["order_hashes"] = {}
    with ZipFile(archives[0]) as archive:
        names = [f"{seed}/image_{i:04d}.png" for seed in (20261111, 20261112) for i in range(22)]
        if sorted(archive.namelist()) != names:
            raise ValueError("Image inventory mismatch")
        for seed in ("20261111", "20261112"):
            hashes = [hashlib.sha256(archive.read(name)).hexdigest() for name in names if name.startswith(seed + "/")]
            orders[seed] = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    output = content / "ns64_clean" / "output"
    output.mkdir(exist_ok=True)
    with TemporaryFile(dir=output) as test:
        test.write(b"output probe")
        test.seek(0)
        if test.read() != b"output probe":
            raise OSError("Output write check failed")
    receipt.update(status="POST_STAGING_CHECKS_PASS", output_writable=True)
except Exception as error:
    receipt["error"] = f"{type(error).__name__}: {error}"
destination = content / "imc_ns64_post_staging.json"
destination.write_text(json.dumps(receipt, indent=2))
print(receipt["status"], receipt.get("error", ""))
