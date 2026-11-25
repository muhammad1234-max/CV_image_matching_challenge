"""Independent administrator-only Haiper scoring, after receipt and parent validation."""

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.novel_model.geometry_capture_v1.database import file_sha, save
from src.novel_model.geometry_capture_v1.parity import compare


def evaluate(data, mapping, output, branch):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf8"))))
    for row in rows:
        row["image"] = mapping[row["image"]]
    pred = output / (branch + "_restored.csv")
    with pred.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "scene",
                "image",
                "rotation_matrix",
                "translation_vector",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image"]))
    result = output / (branch + "_evaluation.json")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/baselines/development/evaluate.py"),
            "imc2023_haiper",
            str(pred),
            str(result),
        ],
        check=True,
    )
    return json.loads(result.read_bytes())["scores"]


def finalize(receipts, validation, output):
    import hashlib

    validation = json.loads(validation.read_bytes())
    if not validation["parent_verified"] or validation["receipt_sha256"] != file_sha(
        receipts
    ):
        raise ValueError("validate this capture first")
    with zipfile.ZipFile(receipts) as z:
        read = lambda n: json.loads(z.read(n))
        m = read("capture/manifest.json")
        if (
            m["backup_sha256"] != validation["parent_sha256"]
            or read("run.json")["status"] != "PASS"
            or read("capture/stage.json")["stage"] != "both_exports_complete"
        ):
            raise ValueError("incomplete two-branch execution")
        inputs = [read(f"capture/{n}_input.json") for n in ("original", "pass_through")]
        expected = dict(
            database_sha256=m["backup_sha256"],
            records=m["original_logical"],
            options=m["mapper_options"],
            permutation_sha256=m["permutation_sha256"],
            environment_receipt_sha256=m["environment_receipt_sha256"],
        )
        if any(v != expected for v in inputs):
            raise ValueError("mapper inputs differ")
        left, right = [
            z.read("capture/" + n + ".csv") for n in ("original", "pass_through")
        ]
        poses = compare(left, right)
        maps = [
            read("capture/" + n + "_map.json") for n in ("original", "pass_through")
        ]
        if not compare(left, z.read("opaque.csv"))["pose_equivalent"]:
            raise ValueError("source exporter differs")
        map_path = ROOT / "reports/phase3c7/private.json"
        mapping = json.loads(map_path.read_bytes())
        # Bind restored identities to the original admin-side neutral image order.
        entries = json.loads(
            (
                ROOT / "configs/baselines/organizer_v1/development_inputs.json"
            ).read_bytes()
        )
        import random

        group = sorted(r["image"] for r in entries if r["dataset"] == "imc2023_haiper")
        # The project's protected mapping is linked to the historical validated restored output.
        if set(mapping.values()) != set(group) or len(mapping) != 54:
            raise ValueError("private membership mismatch")
        with zipfile.ZipFile(ROOT / "reports/phase3c7/worker.zip") as historical:
            hist = historical.read("opaque.csv")
        historical_rows = {
            r["image"]: r for r in csv.DictReader(io.StringIO(hist.decode()))
        }
        hist_path = ROOT / "reports/phase3c7/restored_predictions.csv"
        if (
            file_sha(hist_path)
            != "6901de90ed8cd8cd8b1ab3dd2aba75a986fa64aeb621724b3857f5e91fa4c1c9"
        ):
            raise ValueError("historical restoration pin")
        restored = {
            r["image"]: r for r in csv.DictReader(io.StringIO(hist_path.read_text()))
        }
        for opaque, name in mapping.items():
            if {**historical_rows[opaque], "image": name} != restored[name]:
                raise ValueError("historical private mapping mismatch")
        scores = [
            evaluate(d, mapping, output, n)
            for d, n in ((left, "original"), (right, "pass_through"))
        ]
        same_scores = all(
            abs(v - scores[1][variant][k]) <= 1e-12
            for variant, values in scores[0].items()
            for k, v in values.items()
        )
        parity = (
            poses["same_assignments_and_outliers"]
            and poses["pose_equivalent"]
            and same_scores
            and maps[0]["models"] == maps[1]["models"]
        )
        coverage = validation["coverage"]
        decisions = coverage["methods"]["candidate_a_v1"]["decisions"]
        if not parity:
            status = "BLOCKED — PASS-THROUGH PARITY"
        elif not validation["graph_decisions_reproduced"]:
            status = "BLOCKED — IMPLEMENTATION OR EXECUTION"
        elif "SPLIT" not in decisions:
            status = "BLOCKED — GEOMETRY COVERAGE"
        else:
            status = "READY FOR ACTIVE CANDIDATE A TEST"
        result = dict(
            classification=status,
            parity_pass=parity,
            record_equivalence=True,
            pose_comparison=poses,
            same_scores=same_scores,
            branch_scores=scores,
            branch_mapping=maps,
            historical_comparison=compare(hist, left),
            coverage=coverage,
            resources=read("resources.json"),
            run=read("run.json"),
        )
        save(output / "parity.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipts", "validation", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        print(json.dumps(finalize(a.receipts, a.validation, a.output), indent=2))
    except Exception as exc:
        save(a.output / "failure.json", dict(stage="parity", error=str(exc)))
        raise
