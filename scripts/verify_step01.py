"""Validate audit deliverables, local links and downloaded-source hashes offline."""
import ast
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
checks = []
for name in ["environment_audit", "competition_and_metric", "dataset_access", "baseline_research", "hardware_recommendation"]:
    path = root / "docs" / f"step01_{name}.md"
    assert path.is_file() and path.stat().st_size > 500, path
    checks.append(f"PASS deliverable {path.name}")
for folder in ["docs", "src", "scripts", "tests", "configs", "experiments", "reports", "data"]:
    assert (root / folder).is_dir(), folder
checks.append("PASS all eight scaffold directories")
for path in (root / "scripts").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
checks.append("PASS audit-script Python syntax (no execution of downloaded code)")
for path in [root / "README.md", *(root / "docs").glob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" not in target and not target.startswith("#"):
            assert (path.parent / target.split("#")[0]).exists(), (path, target)
checks.append("PASS internal Markdown links")
manifest_path = root / "docs/evidence/public_sources/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for item in manifest:
    if "sha256" in item:
        body = (manifest_path.parent / item["file"]).read_bytes()
        assert len(body) == item["bytes"] and hashlib.sha256(body).hexdigest() == item["sha256"]
checks.append("PASS downloaded public-source lengths and SHA256 hashes")
for filename in ["environment.json", "kaggle_access.json"]:
    json.loads((root / "docs/evidence" / filename).read_text(encoding="utf-8-sig"))
checks.append("PASS diagnostic JSON parsing")
assert not list((root / "data").rglob("*.csv"))
checks.append("PASS no competition CSV downloaded")
assert not any((root / "src").iterdir())
checks.append("PASS src remains empty: no matching system implemented")
report = "\n".join(checks) + "\n"
(root / "docs/evidence/verification.txt").write_text(report, encoding="utf-8")
print(report)
