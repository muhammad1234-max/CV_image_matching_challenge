"""Read-only listing and allowlisted small metadata download; no submissions."""
import contextlib
import io
import json
from pathlib import Path

competition = "image-matching-challenge-2025-ongoing"
report = {"competition": competition, "files": [], "downloads": []}
buffer = io.StringIO()
try:
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        report["authentication_loaded"] = True
        token = None
        for page in range(100):
            response = api.competition_list_files(competition, page_token=token, page_size=200)
            report["files"].extend({"name": f.name, "bytes": f.total_bytes} for f in response.files)
            token = response.next_page_token
            if not token:
                break
        report["listing_complete"] = not bool(token)
        report["listed_total_bytes"] = sum(int(f["bytes"]) for f in report["files"])
        destination = Path("data/metadata")
        destination.mkdir(parents=True, exist_ok=True)
        for item in report["files"]:
            if item["name"] in {"sample_submission.csv", "train_labels.csv", "train_thresholds.csv"} and int(item["bytes"]) <= 5_000_000:
                api.competition_download_file(competition, item["name"], path=str(destination), quiet=True)
                report["downloads"].append(item["name"])
except Exception as exc:
    # Do not serialize exception bodies, headers, account names, URLs, or tokens.
    report["error_type"] = type(exc).__name__
    response = getattr(exc, "response", None)
    report["http_status"] = getattr(response, "status_code", None)
    report["error_note"] = "Exception body suppressed to prevent credential or signed-URL disclosure."
Path("docs/evidence/kaggle_access.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
print("Listed file count:", len(report["files"]))
