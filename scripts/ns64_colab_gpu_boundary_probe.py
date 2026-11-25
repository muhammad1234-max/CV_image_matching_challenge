"""Synthetic namespace boundary check for a fresh Colab notebook."""
import hashlib
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

PROBE = r'''
import ctypes
import json
import os
import socket
from pathlib import Path
reads = {}
targets = [
    os.environ["CANARY"], "/content/drive", "/content/ns64/project",
    "/content/ns64/data/train/ETs", "/kaggle/input", "/private.json",
    "/input/../../private.json", "/proc/1/root" + os.environ["CANARY"],
    "/input/escape", "/var/run/docker.sock",
]
for name in targets:
    try:
        fd = os.open(name, os.O_RDONLY)
    except OSError as error:
        reads[name] = {"denied": True, "errno": error.errno}
    else:
        os.close(fd)
        reads[name] = {"denied": False}
writes = {}
for name in ("/unexpected", "/input/unexpected"):
    try:
        Path(name).write_text("probe")
    except OSError as error:
        writes[name] = {"denied": True, "errno": error.errno}
    else:
        writes[name] = {"denied": False}
allowed = Path("/input/image_0000.png").read_bytes() == b"synthetic fixture"
Path("/output/write-test").write_text("output writable")
libc = ctypes.CDLL(None, use_errno=True)
class Header(ctypes.Structure):
    _fields_ = [("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class Data(ctypes.Structure):
    _fields_ = [("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
header = Header(0x20080522,0)
data = (Data*2)()
if libc.capget(ctypes.byref(header),ctypes.byref(data)) != 0:
    raise OSError(ctypes.get_errno(), "capget")
controls = {
    "CapEff": f"{data[1].effective:08x}{data[0].effective:08x}",
    "CapPrm": f"{data[1].permitted:08x}{data[0].permitted:08x}",
    "NoNewPrivs": str(libc.prctl(39,0,0,0,0)),
}
bounding = [c for c in range(64) if libc.prctl(23,c,0,0,0) == 1]
ambient = [c for c in range(64) if libc.prctl(47,1,c,0,0) == 1]
interfaces = socket.if_nameindex()
passed = (
    all(v["denied"] for v in reads.values())
    and all(v["denied"] for v in writes.values())
    and allowed and controls["CapEff"] == "0000000000000000"
    and controls["CapPrm"] == "0000000000000000"
    and not bounding and not ambient
    and controls["NoNewPrivs"] == "1"
    and os.getpid() == 1
    and all(name == "lo" for _,name in interfaces)
    and not Path("/proc/self").exists() and not Path("/proc/1").exists()
)
record = dict(
    status="SYNTHETIC_BOUNDARY_PASS" if passed else "BLOCKED",
    reads=reads, writes=writes, permitted_input=allowed, controls=controls,
    uid=os.getuid(), gid=os.getgid(), pid=os.getpid(),
    setup_observations=json.loads(os.environ["SETUP_OBSERVATIONS"]),
    bounding_capabilities=bounding, ambient_capabilities=ambient,
    network_devices=interfaces, python=__import__("sys").version,
)

gpu = {"status": "BLOCKED"}
try:
    cuda = ctypes.CDLL("/runtime/libcuda.so.1")

    gpu["device_access"] = {}
    for path in Path("/dev").glob("nvidia*"):
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError as error:
            gpu["device_access"][path.name] = {"errno":error.errno}
        else:
            os.close(fd)
            gpu["device_access"][path.name] = {"opened":True}
    gpu["nvidia_proc_visible"] = Path("/proc/driver/nvidia/version").is_file()
    result = cuda.cuInit(0)
    for function, key in (("cuGetErrorName","error_name"),("cuGetErrorString","error_description")):
        value = ctypes.c_char_p()
        if getattr(cuda,function)(result,ctypes.byref(value)) == 0 and value.value:
            gpu[key] = value.value.decode(errors="replace")

    gpu["cuInit"] = result
    if result == 0:
        count = ctypes.c_int()
        result = cuda.cuDeviceGetCount(ctypes.byref(count))
        gpu.update(cuDeviceGetCount=result, count=count.value)
        if result == 0 and count.value > 0:
            name = ctypes.create_string_buffer(256)
            code = cuda.cuDeviceGetName(name, len(name), 0)
            gpu.update(cuDeviceGetName=code, name=name.value.decode(errors="replace"))
            if code == 0:
                gpu["status"] = "GPU_ENUMERATION_PASS"
except Exception as error:
    gpu.update(error_type=type(error).__name__, error=str(error))
record["gpu"] = gpu

Path("/output/probe.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
'''

SETUP = r'''
import ctypes
import os
import subprocess
import sys
from pathlib import Path
root, inputs, output, canary = map(Path, sys.argv[1:])
def mount(*args):
    subprocess.run(["mount", *map(str,args)], check=True)
mount("--make-rprivate", "/")
mount("--bind", root, root)
mount("--bind", inputs, root/"input")
mount("-o", "remount,bind,ro,nosuid,nodev", root/"input")
mount("--bind", output, root/"output")

for target in (root/"dev").iterdir():
    mount("--bind", Path("/dev")/target.name, target)

if Path("/proc/driver/nvidia").is_dir():
    mount("--bind", "/proc/driver/nvidia", root/"proc/driver/nvidia")
    mount("-o", "remount,bind,ro,nosuid,nodev,noexec", root/"proc/driver/nvidia")
mount("-o", "remount,bind,ro,nosuid,nodev", root)
import json
observations = json.dumps({
    "mount_namespace":os.readlink("/proc/self/ns/mnt"),
    "network_namespace":os.readlink("/proc/self/ns/net"),
    "mountpoints":[line.split()[4] for line in Path("/proc/self/mountinfo").read_text().splitlines()],
    "observed_by":"trusted setup before chroot",
})
os.chroot(root)
os.chdir("/output")
libc = ctypes.CDLL(None, use_errno=True)
class Header(ctypes.Structure):
    _fields_ = [("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class Data(ctypes.Structure):
    _fields_ = [("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
if libc.prctl(38,1,0,0,0) != 0:
    raise OSError(ctypes.get_errno(), "no-new-privileges")
# Prevent root identity from regaining capabilities across exec.
for capability in range(64):
    if libc.prctl(24, capability, 0, 0, 0) != 0 and ctypes.get_errno() != 22:
        raise OSError(ctypes.get_errno(), "drop capability bounding set")
header = Header(0x20080522,0)
data = (Data*2)()
if libc.capset(ctypes.byref(header),ctypes.byref(data)) != 0:
    raise OSError(ctypes.get_errno(), "capset")
os.execve("/runtime/python",
          ["/runtime/python","-I","/runtime/probe.py"],
          {"PYTHONHOME":"/runtime", "HOME":"/output", "CANARY":str(canary), "PATH":"/runtime",
           "SETUP_OBSERVATIONS":observations})
'''


def copy_libraries(binary, root):
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True)
    for name in re.findall(r"(/[^\s()]+)", result.stdout):
        source = Path(name)
        if source.is_file():
            destination = root/source.relative_to("/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.resolve(), destination)


def main():
    report = {"phase": "synthetic CPU namespace probe", "gpu_readiness": "driver enumeration probe only"}
    if sys.platform != "linux" or not shutil.which("unshare"):
        raise RuntimeError("Requires Linux with unshare")
    report["host_python"] = str(Path(sys.executable).resolve())
    with tempfile.TemporaryDirectory(prefix="ns64_namespace_") as temporary:
        base = Path(temporary)
        root, inputs, output = (base/n for n in ("root","input","output"))
        for path in (root,inputs,output):
            path.mkdir()
        canary = base/"private-admin-canary"
        canary.write_text("public isolation marker")
        report["host_canary_exists"] = canary.is_file()
        (inputs/"image_0000.png").write_bytes(b"synthetic fixture")
        (inputs/"escape").symlink_to(canary)
        for name in ("runtime","runtime/lib","input","output","proc","proc/driver","proc/driver/nvidia","dev"):
            (root/name).mkdir(exist_ok=True)
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        stdlib = Path(sysconfig.get_path("stdlib"))
        shutil.copytree(stdlib, root/"runtime/lib"/version,
                        ignore=shutil.ignore_patterns("site-packages","dist-packages","__pycache__","test","tests","ensurepip","idlelib","tkinter"))
        executable = Path(sys.executable).resolve(strict=True)
        shutil.copy2(executable,root/"runtime/python")

        copy_libraries(executable,root)
        listing = subprocess.run(["ldconfig","-p"],capture_output=True,text=True).stdout
        candidates = [line.split("=>",1)[1].strip() for line in listing.splitlines()
                      if "libcuda.so.1 " in line and "=>" in line]
        directories = [
            "/usr/lib64-nvidia", "/usr/local/nvidia/lib64",
            "/usr/lib/x86_64-linux-gnu", "/usr/lib/wsl/lib",
            *os.environ.get("LD_LIBRARY_PATH","").split(":"),
        ]
        candidates.extend(str(Path(d)/"libcuda.so.1") for d in directories if d)
        loaded = None
        errors = []
        for name in ["libcuda.so.1", *dict.fromkeys(candidates)]:
            try:
                loaded = ctypes.CDLL(name)
                break
            except OSError:
                errors.append(name)
        mapped = []
        if loaded is not None:
            for line in Path("/proc/self/maps").read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts)==6 and Path(parts[5]).name.startswith("libcuda.so"):
                    mapped.append(Path(parts[5]))
        if not mapped:
            report.update(error="CUDA driver library discovery failed",
                          searched=errors)
            destination = Path.cwd()/"imc_ns64_colab_gpu_boundary_probe.json"
            destination.write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
            return
        driver = mapped[0].resolve(strict=True)
        report["driver_library"] = str(driver)
        report["driver_library_sha256"] = hashlib.sha256(driver.read_bytes()).hexdigest()
        shutil.copy2(driver,root/"runtime/libcuda.so.1")
        copy_libraries(driver,root)
        import stat
        for device in Path("/dev").glob("nvidia*"):
            if stat.S_ISCHR(device.stat().st_mode):
                (root/"dev"/device.name).touch()
        for name in ("null","urandom","random","zero"):
            (root/"dev"/name).touch()
        report["host_gpu"] = subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],text=True).strip()

        # Resolve shared libraries used by the standard-library extension modules.
        for extension in (root/"runtime/lib"/version/"lib-dynload").glob("*.so"):
            copy_libraries(extension,root)
        (root/"runtime/probe.py").write_text(PROBE)
        setup = base/"setup.py"
        setup.write_text(SETUP)
        command = ["unshare","--user","--map-root-user","--mount","--pid","--fork","--net",
                   str(executable),"-I",str(setup),str(root),str(inputs),str(output),str(canary)]
        try:
            run = subprocess.run(command,capture_output=True,text=True,timeout=45,
                                 cwd="/",close_fds=True,
                                 env={"PATH":"/usr/sbin:/usr/bin:/sbin:/bin","LANG":"C.UTF-8"})
            report.update(exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr)
            if (output/"probe.json").is_file():
                report["worker"] = json.loads((output/"probe.json").read_text())
        except subprocess.TimeoutExpired:
            report.update(exit_code=None,error="namespace probe timeout")
    destination = Path.cwd()/"imc_ns64_colab_gpu_boundary_probe.json"
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print("Saved",destination.name)


if __name__ == "__main__":
    main()
