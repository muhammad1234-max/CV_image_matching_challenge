"""Package the gate status and operator artifacts without numerical parents or private data."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from scripts.phase5a7_build import R,OLD,ORIGINAL,JOURNALS,build
from scripts.phase5a7_recovery import LOST,ARCHIVE,DEBIT,ATTEMPT
from src.runner_5a5.kaggle import sha,save

def pack():
    fixture=build()
    transfer=json.loads((R/"workstation_transfer.json").read_bytes())
    assert transfer["release"]==fixture and transfer["status"]=="FIXTURE_RESTORE_PASS" and transfer["bytes"]>=1195338874
    xml=ET.parse(R/"release_tests.xml").getroot().find("testsuite")
    assert all(xml.get(k)=="0" for k in ("failures","errors","skipped"))
    tests=int(xml.get("tests"))
    ops=[Path(p) for p in ("scripts/phase5a7_recovery.py","src/runner_5a7_primitives.py","scripts/phase5a7_persistence.py")]
    descriptor=dict(parent_release=OLD,fixture_release=fixture,changed_files={p.as_posix():sha(p) for p in ops})
    revision=hashlib.sha256(json.dumps(descriptor,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    save(R/"recovery_extension.json",dict(**descriptor,extension_sha256=revision,state="prepared; live dispatch withheld at persistence gate"))
    ledger=dict(cumulative_limit=14400,historical_capture_complete=True,historical_seal=LOST,
        historical_archive_expected=ARCHIVE,consumed=DEBIT,remaining=14400-DEBIT,trial_limit=1,
        recovery_capture_allowance=1,recovery_captures_started=0,recovery_seconds=None,recovery_overhead_seconds=None,
        latest_journal=JOURNALS[-1],attempt_id=ATTEMPT,active_parent=None,
        recovery_transition="specified and tested; uncommitted",new_journal=None,
        remote_preflight="pending external custody proof",original="not started",trial="not started",
        decision=None,score=None,paired_delta=None,classification="BLOCKED — PERSISTENCE OR CHECKPOINT CUSTODY")
    save(R/"ledger.json",ledger)
    audit=dict(local="Project search and workstation Downloads/Desktop found no Dioscuri capture archive.",
        runtime="Operator reported archive absent, sealed parent absent and runner unloaded.",
        notebook_versions="Operator reported no saved version for the original run.",
        private_inventory="Local authenticated CLI request to list owned notebooks returned Authentication required. Operator-side private dataset/output check outstanding.",
        other_destinations="Phase 5A.6 transport targeted the browser download. No additional configured persistent destination identified.",
        observed_failure="Export hash printed; no retained matching archive. Runtime disappeared before an external read-back. Browser/network failure mechanism unobserved.",
        recommended_destination="A completed private Kaggle Save & Run All version, verified from a fresh reader.",
        platform_documentation="https://www.kaggle.com/docs/notebooks",
        documented_output_capacity="Up to 20 GB; actual private version completion and access still require live proof.",
        representative_target_bytes=1195338874,workstation_archive_bytes=transfer["bytes"],remote_persistence="unverified")
    save(R/"custody_audit.json",audit)
    docs={
    "recovery_authorization":f"""Supervisor prompt 37 permits one replacement CAPTURE for the same 138 opaque Dioscuri/Baalshamin images, dev_a, heritage_dioscuri, seed 20261111. The latest authoritative journal is {JOURNALS[-1]}. The original capture remains complete. Its debit is {DEBIT} seconds. The remaining budget is {14400-DEBIT} seconds. The authorization has not been consumed. No scientific setting changed.""",
    "custody_failure_analysis":"""Phase 5A.6 validated the local export and printed its SHA-256 before prompting for browser custody. The browser failed to deliver a retained copy. The operator later found neither the .bin nor the sealed directory or process in the runtime and reported no saved version. The available evidence cannot distinguish a browser, network, session or service-side transfer error. No verified usable parent has been found. The API rejected the private inventory request. Before further GPU work the owner must also check private notebook outputs, datasets and other destinations. If the original is found, validate its SHA-256 and restore it with the original contract and completion journal. A filename match is insufficient.""",
    "persistence_runbook":f"""Only CPU is needed for these two notebooks. Keep both private and turn Internet off.

1. Check Your Datasets and Your Work for a previously retained imc2024_dioscuri_baalshamin_capture.bin. If found, preserve it and return its version reference for verification. Otherwise import notebooks/step05a7_custody_producer.ipynb in a new private notebook. Attach no inputs. Choose Save Version > Save & Run All. Allow the committed version to finish.
2. Open the completed version's Output tab and confirm phase5a7_fixture_capture.bin, phase5a7_fixture_journal.json and phase5a7_producer.json exist. Keep this version. Record its notebook name, version number, private visibility and the producer receipt SHA256 from the completed version's log. The large .bin stays on Kaggle. If the version fails or outputs are absent, retain the log and stop.
3. Import notebooks/step05a7_custody_reader.ipynb into a second fresh private CPU notebook. Under Add Input select the first notebook's saved output. Attach only that version. Run the cell interactively and paste the SHA256 from step 2. On FIXTURE_RESTORE_PASS, download the small phase5a7_external_restore.json. Download the producer's small phase5a7_producer.json too. Return these JSONs, the producer version reference, and confirmation of private visibility. Keep the version accessible. A browser receipt alone does not preserve the checkpoint.
4. On the trusted workstation reconcile the proof, external head, and private inventory before issuing any recovery dispatch. Each later real stage needs an independently retained start record plus an externally verified completed archive and journal. An unattended replay of Phase 5A.6 would repeat matching. Use only a future recovery invocation bound to authenticated custody. Never release a second replacement after an interrupted start.

The producer uses the Phase 5A.6 production contract on disposable data. The transport test uses incompressible synthetic SQLite blocks with a target of 1,195,338,874 bytes, the prior live archive planning allowance. Local receipt size was {transfer['bytes']:,} bytes. The actual lost archive length was never established. The local rehearsal used separate interpreters; the remote private version and fresh reader are still required. Kaggle documents up to 20 GB saved output and notebook-output reuse. The reader tests actual access and capacity for this fixture. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)""",
    "journal_recovery_contract":f"""The retained snapshots form a byte-verified append-only progression. The proposed recovery transition carries the original journal, historical seal and expected archive digest, reason, allowance, attempt ID, prior debit, remaining budget and new release. It is uncommitted pending the persistence proof. Extension descriptor {revision} derives from {OLD}. The CPU persistence notebooks use their own artifact identity {fixture}.

The administrator's single-successor registry lives outside the producing runtime. Compare-and-append prevents accepting a stale branch. The historical debit remains in the single cumulative ledger. Only stage repetition and active registration are scoped to the replacement generation. The production Runner still validates its stage contract at seal, export and restore. A new attempt is recorded before dispatch. An unresolved start keeps its reserved time encumbered and blocks another launch; its exact duration stays unknown pending reconciliation. A distinct completion journal and a read-only retrieved version are required for parent activation. The restored numerical database binds both mappers. Never recreate an administrator registry from a stale mounted journal.""",
    "recovery_tests":f"""The final local suite passed {tests} tests. The full XML and log are included. The duplicate-member ZIP fixture may emit its expected warning. Recovery admission tests cover available or unsearched destinations, missing authorization and persistence proof, and absent historical archives (A–D). History, debit, reopening, stale heads, single use and interruption tests cover E–I. Foreign identity, missing outputs, synthetic archives, checksum mismatches and the historical seal exercise J–M. Activation, external-mount checks, parent-bound branch execution, digest failures and timeout fallback cover N–Q. The build independently verified 45 scientific hashes, including the evaluator (R). The two Kaggle notebooks were rehearsed locally in separate interpreters. The local state-machine tests substitute the external destination; only a successful private version retrieval can satisfy the live gate.""",
    "live_preflight":"""Phase 5A.6 passed Kaggle production preflight on the prior T4 allocation. The recovery has no new live GPU preflight. CPU custody proof is pending. The subsequent live gate must measure the pinned runtime, native dependencies, selected GPU UUID and free VRAM, approved inputs, source, weights, cgroup charge and OOM counters, pressure, scratch, export space and session time. Use the minimum of actual time and the cumulative balance. Preserve the 8,400-second capture floor, 1,200-second reserve and 5,400-second trial ceiling.""",
    "capture_results":f"""The historical CAPTURE completed in {DEBIT} seconds with seal {LOST}. Its reported archive SHA256 is {ARCHIVE}. Independent retrieval is unavailable. Replacement attempts started: zero. Recovery runtime, parent SHA256, new seal, archive and durable version are unavailable.""",
    "original_results":"ORIGINAL awaits an active, independently restored numerical parent. No real original reconstruction, registration coverage, clusters, poses or prediction file exists.",
    "trial_results":"TRIAL has never started. The one scientific trial remains available subject to frozen eligibility, resources and a sealed ORIGINAL. No proposal or trial measurement exists.",
    "automatic_decision":"The selector has no sealed inputs. ACCEPT, REJECT and ABSTAIN have not been evaluated.",
    "independent_scoring":"Prospective scores and paired differences are unavailable. Historical registered organizer scores are mAA 0.4097222222222222, clusterness 0.898876404494382 and combined 0.5628763937749686 in both frozen variants. Same-parent original, trial and automatic comparisons await validated predictions. No normalized-cut row was scored, and no labels were accessed for selection. Report the historical lost capture debit separately from normal method runtime and measured recovery overhead.",
    "supervisor_handoff":f"""BLOCKED — PERSISTENCE OR CHECKPOINT CUSTODY

The original captured numerical parent was lost after {DEBIT} seconds. The completion history is intact and charged once. One recovery is authorized and unused. The available balance is {14400-DEBIT} seconds. The worker, configuration, protocol, image archive, image order and frozen scientific identities were verified from available bytes. Verified identities and logs are in reports/phase5a7. The recovery extension descriptor is {revision}; the previous production release was {OLD}. No new prospective journal has been committed.

The workstation representative transfer and full restore passed. The private remote destination and its permissions await the CPU saved-version experiment. No recovery GPU execution, new parent, active seal, original, trial, automatic decision or prospective score exists. Run the producer and reader described in the persistence runbook. Return both JSON receipts and the private version reference. If remote proof succeeds, complete the production dispatch binding and live preflight before issuing the single recovery. No further scientific authorization is required for the existing scope. If the original archive surfaces, recover that historical parent first."""
    }
    paths=[]
    for name,body in docs.items():
        p=Path("docs/phase5a7_"+name+".md")
        p.write_text("# "+name.replace("_"," ").capitalize()+"\n\n"+body+"\n",encoding="utf8")
        paths.append(p)
    paths+=[*Path("scripts").glob("phase5a7_*.py"),ops[1],Path("tests/test_phase5a7_recovery.py"),
            *Path("notebooks").glob("step05a7*.ipynb"),Path("phase5a6_capture_recovery_review.zip"),
            *R.glob("*.json"),R/"release_tests.xml",R/"release_tests.log"]
    hashes={p.as_posix():sha(p) for p in paths}
    dest=Path("phase5a7_dioscuri_recovery_review.zip")
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.as_posix())
        z.writestr("SHA256.json",json.dumps(hashes,indent=2,sort_keys=True))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in hashes.items())
    print(dest,sha(dest),tests)
if __name__=="__main__":pack()
