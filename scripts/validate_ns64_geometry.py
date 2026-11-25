"""Administrator-side numerical parent verification. Never import scoring data here."""

import argparse
import io
import json
from contextlib import closing
import sys
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.novel_model.geometry_capture_v1.database import (
    file_sha,
    sha,
    save,
    inventory,
    logical,
    read_only,
    OFFSET,
)
from src.novel_model.geometry_capture_v1.analysis import convert


def verify_capture(receipts, parent, output):
    read = lambda p: json.loads((ROOT / p).read_bytes())
    prep = read("reports/phase3c7/preparation.json")
    pins = read("reports/phase4c/workflow_pins.json")
    for name, h in pins["adapter_files"].items():
        if file_sha(ROOT / name) != h:
            raise ValueError("adapter mismatch " + name)
    with zipfile.ZipFile(receipts) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("duplicate receipt")
        load = lambda n: json.loads(z.read(n))
        m = load("capture/manifest.json")
        env = load("environment.json")
        run = load("run.json")
        if (
            env["status"] != "CLEAN_VM_PRECHECK_PASS"
            or env["unexpected_artifact_count"] != 0
            or load("boundary.json")["status"] != "PASS"
            or set(load("weight_load.json")["loaded"]) != set(prep["checkpoint_hashes"])
            or m["environment_receipt_sha256"] != sha(z.read("environment.json"))
        ):
            raise ValueError("boundary, weights or environment receipt mismatch")
        with zipfile.ZipFile(ROOT / "reports/phase3c7/worker.zip") as ref:
            hist = json.loads(ref.read("environment.json"))
        for k in (
            "environment_freeze",
            "torch",
            "pycolmap",
            "device",
            "source_inventory_hash",
            "assets",
        ):
            if env[k] != hist[k]:
                raise ValueError("environment drift: " + k)
        expected = dict(
            config_sha256=prep["config_sha256"],
            source_revision=prep["source_revision"],
            source_inventory_hash=prep["source_inventory_hash"],
            checkpoints=prep["checkpoint_hashes"],
            adapter_bundle_sha256=pins["adapter_bundle_sha256"],
            worker_sha256=pins["capture_worker_sha256"],
            permutation_sha256=prep["permutation_hash"],
            schema="ns64_pre_mapper_v1",
            stage="complete_pre_mapper_capture",
            coordinates=OFFSET,
            seed=20261111,
        )
        if (
            any(m.get(k) != v for k, v in expected.items())
            or run["worker_sha256"] != expected["worker_sha256"]
        ):
            raise ValueError("capture identity mismatch")
        names = [f"image_{i:04}.png" for i in range(54)]
        with zipfile.ZipFile(ROOT / "imc2023_haiper_opaque.zip") as images:
            if sorted(images.namelist()) != names:
                raise ValueError("image inventory mismatch")
            authentic = {n: sha(images.read(n)) for n in names}
            if (
                authentic != m["image_sha256"]
                or sha("\n".join(authentic.values()).encode())
                != prep["permutation_hash"]
            ):
                raise ValueError("input/permutation mismatch")
            if file_sha(parent) != m["backup_sha256"]:
                raise ValueError("database digest mismatch")
            with closing(read_only(parent)) as db:
                if logical(db) != m["original_logical"]:
                    raise ValueError("database records mismatch")
            ims, cams, keys, pairs, arrays = inventory(
                parent, names, {int(i): v for i, v in m["offset_sha256"].items()}
            )
            if (
                {str(i): v for i, v in ims.items()} != m["images"]
                or {str(i): v for i, v in cams.items()} != m["cameras"]
                or arrays != m["arrays"]
            ):
                raise ValueError("numerical manifest mismatch")
            for im in ims.values():
                cam = cams[im["camera_id"]]
                n = im["name"]
                with Image.open(io.BytesIO(images.read(n))) as image:
                    eq = image.getexif().get(41989)
                    quality = bool(
                        eq
                        and np.isclose(
                            float(eq) / 35 * max(image.size),
                            cam["params"][0],
                            rtol=1e-5,
                        )
                    )
                    if quality != m["focal_quality"][n] or image.size != (
                        cam["width"],
                        cam["height"],
                    ):
                        raise ValueError("calibration provenance")
        # Independent reimport. Cross-platform numerics need not yield bit-identical residuals.
        stats = convert(parent, m, output)
        remote = load("capture/coverage.json")
        decisions_match = (
            stats["methods"] == remote["methods"]
            and stats["reasons"] == remote["reasons"]
        )
        result = {
            "parent_verified": True,
            "parent_sha256": m["backup_sha256"],
            "receipt_sha256": file_sha(receipts),
            "remote_run_status": run["status"],
            "graph_decisions_reproduced": decisions_match,
            "coverage": stats,
        }
        save(output / "validation.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("receipts", "parent", "output"):
        parser.add_argument(n, type=Path)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        result = verify_capture(a.receipts, a.parent, a.output)
    except Exception as exc:
        save(
            a.output / "failure.json", {"stage": "validate_capture", "error": str(exc)}
        )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
