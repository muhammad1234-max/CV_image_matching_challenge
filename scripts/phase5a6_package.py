"""Package the verified initial release and externally retained migration receipt."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from scripts.phase5a6_build import build,R,ADAPTER,read,sha
from src.runner_5a5.kaggle import JOURNAL,transition,save
from src.runner_5a1.runner import Runner
from scripts.phase5a6_production import Contract

def counts(path):
    node=ET.parse(path).getroot().find("testsuite")
    assert all(node.get(k)=="0" for k in ("failures","errors","skipped"))
    return int(node.get("tests"))

def pack():
    payload=build()
    release=read(R/"release.json")
    rev=release["production_release"]
    fixture_count=counts(R/"release_dispatch.xml")
    full=counts(R/"full.xml")
    prior=Path("phase5a5_supervisor_review.zip")
    def verify(z,manifest):
        pins=json.loads(z.read(manifest))
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in pins.items())
        return len(pins)
    with zipfile.ZipFile(prior) as outer:
        outer_count=verify(outer,"SHA256.json")
        with zipfile.ZipFile(io.BytesIO(outer.read("phase5a5_kaggle_migration_review.zip"))) as inner:
            inner_count=verify(inner,"reports/phase5a5/SHA256.json")
    assert (outer_count,inner_count)==(4,32)
    source=Path("journal-"+JOURNAL+".json")
    contract=Contract(release["pins"],rev)
    custody=Path("kaggle_upload/phase5a6_initial_custody")
    custody.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=R) as tmp:
        d=Path(tmp)
        path=d/"attempts.json"
        transition(source,path,contract.identity,rev)
        events=read(path)
        events[-1]["reason"]="Phase 5A.6 approved Kaggle production migration"
        save(path,events)
        pin=sha(path)
        runner=Runner(d,contract,lambda *a:None,pin)
        assert runner.events[-1]["previous_events"]==read(source) and len(runner.events)==2
        saved=custody/("journal-"+pin+".json")
        assert all(p==saved for p in custody.iterdir()),"existing custody differs"
        if not saved.exists():
            with saved.open("xb") as f:f.write(path.read_bytes())
        assert sha(saved)==pin
    historical=read("phase5a5_kaggle_smoke.json")["checks"]
    organizer=read("experiments/baseline/organizer_baseline_v1_dev_full.json")["datasets"]["imc2024_dioscuri_baalshamin"]["scores"]
    status=dict(classification="BLOCKED — RESOURCES OR EXECUTION",release=rev,adapter=ADAPTER,
        registered_organizer=organizer,committed_journal=pin,prior_journal=JOURNAL,
        approved_images=138,partition="dev_a",stage_seconds_consumed=0,stage_seconds_remaining=14400,
        trial_allowance_remaining=1,real_checkpoints=[],preflight="AWAIT_LIVE_KAGGLE",capture="NOT_STARTED",
        original="NOT_STARTED",trial="NOT_STARTED",decision=None,candidate_score=None,paired_difference=None,
        scientific_files=len(read(R/"scientific.json")),tests=dict(previous_reference=406,local_full=full,
        final_packaged_dispatch=fixture_count,packaged_regressions=counts(R/"packaged_fixture.xml")),
        prior_review_manifest_matches=dict(outer=outer_count,inner=inner_count))
    save(R/"status.json",status)
    docs={
      "kaggle_production_integration":f"""The production archive digest is {rev}. The notebook verifies it before extraction. The approved adapter is {ADAPTER}. The independent scorer stays on the workstation. The release reuses the approved checkpoint Runner and branch provenance contract, binds its generated programs to the measured runtime and GPU, and traces resources in the process tree. Capture validation also requires numerical raw and verified matches. Failed indexed exports block sealing.

Local regression tests passed ({full} tests, including the duplicate-ZIP warning). The final packaged dispatch tests passed ({fixture_count} cases). They exercise generated programs, native-service substitutes, receipt transfer, full checkpoint export and restore, rejection, timeout fallback and a digest failure. Windows uses its local interpreter. Kaggle additionally runs the fixture subprocesses with the pinned interpreter and resource monitor. A separate generated preamble checks the actual runtime before matching. The first production notebook session still needs to run those gates.""",
      "live_preflight":"""No Phase 5A.6 live measurement has been returned. Phase 5A.5 measured Python 3.11.11, PyTorch 2.5.1+cu121, a single visible Tesla T4, a 30 GiB cgroup and 19.50 GiB of export space after setup. Its estimated scratch requirement was 6.79 GiB. Production measures scratch and export separately, reads OOM events and pressure, selects GPU 0 by its current UUID, checks the second GPU and uses the operator's remaining session time. The downloaded preflight JSON records each completed check. The runtime uses clean-environment isolation.""",
      "journal_custody":f"""Local reconciliation found the externally retained created-only journal {JOURNAL}. The committed migration journal is {pin}, under kaggle_upload/phase5a6_initial_custody. Its first event binds release {rev}. Its second preserves the original event, runner, hash, empty checkpoint inventory and zero seconds consumed. Both files are included for audit. Confirm that no newer journal is held elsewhere before using the initial migration. After starting a stage, use only the latest retained journal. An unresolved start or failure requires review.""",
      "capture_results":"CAPTURE awaits live preflight. There is no numerical parent, capture seal, matching runtime or reusable real checkpoint. The next permitted stage is CAPTURE.",
      "original_results":"ORIGINAL awaits a verified, sealed numerical parent. No same-run reference predictions, registration counts or pose metrics are available.",
      "trial_results":"TRIAL awaits ORIGINAL. The trial allowance remains one. Boundary eligibility, pair mask, original and difficult observation retention, and trial costs have no measurements.",
      "automatic_decision":"DECISION awaits verified predecessors. There is no ACCEPT, REJECT or ABSTAIN decision and no sealed automatic prediction.",
      "independent_scoring":f"""The registered organizer scores for this group are {json.dumps(organizer,sort_keys=True)}. These are historical registered values. Recompute them with the frozen evaluator after authenticated transfer. The original ns64, trial, normalized-cut and automatic scores are unavailable in this release. The same-run primary paired difference is pending. No extra trial is authorized for a comparator.

On the trusted workstation retain the four stage .bin files, final preflight JSON and latest journal in one returns directory. Create pins.json from the four independently saved archive digests keyed by capture, original, trial and decision. Run python -m scripts.receive_dioscuri_5a6 RETURNS FINAL_JOURNAL_SHA RETURNS/pins.json NEW_OUTPUT_DIR. The entry point revalidates the stages, freeze and predecessor chain before using the frozen scorer. Keep identity mappings and identity-restored prediction CSVs private.""",
      "scientific_interpretation":"Dioscuri/Baalshamin is one development group with 138 images. No experimental predictions, paired differences, pose or clustering changes, or runtime overhead have been measured. The frozen baseline and validation split remain the required comparators.",
      "supervisor_handoff":f"""BLOCKED — RESOURCES OR EXECUTION

The local production package and journal transition are ready. All {status['scientific_files']} frozen file identities passed local verification during packaging. The prior GPU smoke and restore receipts cover the compatibility release. Current GPU quota, remaining VM time, memory and disk await measurement. Real stage use is zero of 14,400 seconds. The capture, original, trial, automatic decision and paired score are pending. No real numerical checkpoint exists.

Operator procedure

1. Create a private Kaggle dataset from only the single JSON file inside kaggle_upload/phase5a6_initial_custody. Keep its hash-based filename. Confirm there is no later externally retained journal. Keep this dataset separate from the two existing approved inputs.
2. Import notebooks/step05a6_dioscuri_kaggle.ipynb into a fresh private interactive Kaggle notebook. Select T4 x2 and enable internet for pinned setup. Attach only ns64_runtime.bin, dioscuri_opaque.bin and the custody JSON. Detach the competition, the synthetic checkpoint and previous output datasets.
3. Run PREFLIGHT only. Enter the actual remaining session seconds and the latest journal SHA256. For the first run that SHA256 is {pin}. Allow time for setup plus at least 8,400 seconds at capture start. Download phase5a6_preflight.json. On failure also download phase5a6_blocker.json.
4. After PREFLIGHT PASS, run CAPTURE once. For every custody prompt download the named file from /kaggle/working, verify its SHA256 locally with Get-FileHash -LiteralPath 'path-to-downloaded-file' -Algorithm SHA256, and paste the verified digest. Save the started journal before computation, then the completed journal and the stage archive. Keep every earlier archive and journal externally.
5. Continue ORIGINAL, TRIAL and DECISION in that order if the gates permit. Use a single stage cell at a time. On a fresh VM mount only the two approved inputs, the latest journal JSON and each completed stage .bin. Supply independently retained digests. PREFLIGHT restores and validates the complete chain. Run the next unfinished stage. If an archive export needs repetition within the same session, its stage cell verifies the existing seal before exporting. After interruption, preserve all available evidence and stop for review if the latest journal has an unresolved attempt.
6. Return each .bin, the journal files, independently retained hashes, preflight JSON and any blocker or diagnostic files to the trusted workstation. Keep the large archives separately from the supervisor review. Stop after Dioscuri/Baalshamin."""
    }
    paths=[]
    for title,body in docs.items():
        p=Path("docs/phase5a6_"+title+".md")
        p.write_text("# "+title.replace("_"," ").capitalize()+"\n\n"+body+"\n",encoding="utf8")
        paths.append(p)
    paths += [source,saved,*sorted(R.glob("*.json")),*sorted(R.glob("*.xml")),*sorted(R.glob("*.log")),
         *sorted(Path("scripts").glob("phase5a6_*.py")),Path("scripts/receive_dioscuri_5a6.py"),
         Path("tests/test_phase5a6_dispatch.py"),Path("notebooks/step05a6_dioscuri_kaggle.ipynb"),
         Path("phase5a5_kaggle_smoke.json"),Path("phase5a5_kaggle_restore.json"),Path("reports/phase5a5/full.log")]
    target=Path("phase5a6_dioscuri_kaggle_prospective_review.zip")
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.as_posix())
        z.writestr("SHA256.json",json.dumps({p.as_posix():sha(p) for p in paths},sort_keys=True,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and verify(z,"SHA256.json")==len(paths)
    print(saved,rev,target,sha(target))
if __name__=="__main__":pack()
