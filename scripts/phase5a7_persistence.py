"""Disposable CPU fixture for a private, completed Kaggle version and an independent reader."""
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import uuid
from scripts.phase5a6_production import Contract
from src.runner_5a1.contract import Fatal, check, equal, read, save
from src.runner_5a1.runner import Runner
from src.runner_5a5.kaggle import sha
from src.novel_model.geometry_capture_v1.database import logical
from test_phase5a1 import Fixture

MIB = 1024 ** 2
TARGET = 1195338874  # prior live planning allowance for one archive
RECEIPT = "phase5a7_producer.json"
ARCHIVE = "phase5a7_fixture_capture.bin"
JOURNAL = "phase5a7_fixture_journal.json"

def produce(output, revision, target=TARGET):
    output=Path(output)
    for name in (RECEIPT,ARCHIVE,JOURNAL):
        check(not (output/name).exists(),"existing producer artifacts")
    check(shutil.disk_usage(output).free >= 4*target,"insufficient space for export round trip")
    with tempfile.TemporaryDirectory() as t:
        scratch=Path(t)
        check(shutil.disk_usage(scratch).free >= 4*target,"insufficient fixture scratch")
        fixture=Fixture(scratch)
        contract=Contract(fixture.p,revision)
        run=Runner(scratch/"producer",contract,lambda *args:None,"NEW")
        def capture(stage,d,end):
            fixture.child(stage,d,end)
            with closing(sqlite3.connect(d/"parent.db")) as db:
                db.execute("alter table matches add column rows integer")
                db.execute("alter table matches add column data blob")
                db.execute("insert into matches select pair_id,rows,data from two_view_geometries")
                db.execute("create table transfer_blocks(payload blob not null)")
                for _ in range(math.ceil(target/MIB)):
                    db.execute("insert into transfer_blocks values(randomblob(?))",(MIB,))
                db.commit()
                records=logical(db)
            job=read(d/"job.json")
            job["parent"]=dict(original_logical=records,backup_sha256=sha(d/"parent.db"))
            save(d/"job.json",job)
        run.run("capture",capture,time.monotonic()+14400)
        archive=run.export("capture",output/ARCHIVE)
        shutil.copyfile(run.journal,output/JOURNAL)
        receipt=dict(scope="disposable synthetic persistence",release=revision,nonce=uuid.uuid4().hex,
            archive=archive,bytes=(output/ARCHIVE).stat().st_size,journal=run.pin,
            seal=run.verify("capture",run.root/"capture"),parent=sha(run.root/"capture/parent.db"))
        check(receipt["bytes"]>=target,"transfer did not meet planning allowance")
        save(output/RECEIPT,receipt)
        return sha(output/RECEIPT)

def verify(source,receipt_pin,revision,target=TARGET):
    source=Path(source)
    equal(sha(source/RECEIPT),receipt_pin,"independently retained producer receipt")
    receipt=read(source/RECEIPT)
    equal(receipt["release"],revision,"fixture release")
    equal(receipt["scope"],"disposable synthetic persistence","fixture scope")
    archive=source/ARCHIVE
    check(archive.stat().st_size==receipt["bytes"] and receipt["bytes"]>=target,"transfer length")
    equal(sha(source/JOURNAL),receipt["journal"],"external journal")
    with tempfile.TemporaryDirectory() as t:
        scratch=Path(t)
        fixture=Fixture(scratch)
        dest=scratch/"reader"
        dest.mkdir()
        shutil.copyfile(source/JOURNAL,dest/"attempts.json")
        run=Runner(dest,Contract(fixture.p,revision),lambda *args:None,receipt["journal"])
        equal([(e["status"],e["stage"]) for e in run.events],
              [("created",""),("started","capture"),("complete","capture")],"fixture journal progression")
        # Exercise both wrong external pins and genuine structural corruption.
        try:
            run.restore("capture",archive,"0"*64)
        except Fatal:
            pass
        else:
            raise AssertionError("stale digest accepted")
        import zipfile
        incomplete=scratch/"incomplete.bin"
        with zipfile.ZipFile(incomplete,"w") as z:
            z.writestr("seal.json","{}")
        try:
            run.restore("capture",incomplete,sha(incomplete))
        except Fatal:
            pass
        else:
            raise AssertionError("incomplete archive accepted")
        seal=run.restore("capture",archive,receipt["archive"])
        equal(seal,receipt["seal"],"restored seal")
        equal(sha(dest/"capture/parent.db"),receipt["parent"],"restored database")
        equal(run.parents("original"),{"capture":seal},"next stage parents")
        equal(run.pin,receipt["journal"],"restored journal")
        return dict(status="FIXTURE_RESTORE_PASS",producer_sha256=receipt_pin,release=revision,
                    scope=receipt["scope"],archive=receipt["archive"],journal=run.pin,
                    bytes=receipt["bytes"],seal=seal,parent=receipt["parent"],next_stage="original",
                    stale_rejected=True,incomplete_rejected=True)
