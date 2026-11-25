"""Fetch bounded public text sources only; never execute downloaded code."""
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "first_metric.py.txt": "https://raw.githubusercontent.com/ns-rokuyon/kaggle-image-matching-challenge-2025/main/utils/imc25/metric.py",
    "first_report.md": "https://raw.githubusercontent.com/ns-rokuyon/kaggle-image-matching-challenge-2025/main/REPORT.md",
    "first_pyproject.toml": "https://raw.githubusercontent.com/ns-rokuyon/kaggle-image-matching-challenge-2025/main/pyproject.toml",
    "first_evaluate.py.txt": "https://raw.githubusercontent.com/ns-rokuyon/kaggle-image-matching-challenge-2025/main/evaluate_imc2025.py",
    "reference_metric.py.txt": "https://raw.githubusercontent.com/AymenLass/Image-Matching-Challenge-2025-7th-place-solution/main/imc25metric.py",
    "seventh_requirements.txt": "https://raw.githubusercontent.com/AymenLass/Image-Matching-Challenge-2025-7th-place-solution/main/requirements.txt",
    "first_tree.json": "https://api.github.com/repos/ns-rokuyon/kaggle-image-matching-challenge-2025/git/trees/main?recursive=1",
}
destination = Path("docs/evidence/public_sources")
destination.mkdir(parents=True, exist_ok=True)
manifest = []
for filename, url in SOURCES.items():
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "IMC-Step1-Audit"}), timeout=20) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("Source exceeds 2 MB safety cap")
        (destination / filename).write_bytes(body)
        manifest.append({"file": filename, "url": url, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    except Exception as exc:
        manifest.append({"file": filename, "url": url, "error": str(exc)})
print(json.dumps(manifest, indent=2))
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
