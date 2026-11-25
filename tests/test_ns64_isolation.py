from pathlib import Path
import hashlib
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/baselines/top10_reproduction'),str(ROOT/'scripts')]
import isolated_staging as staging
import restore_and_score as scoring
import verify_ns64_isolation as launcher


def test_frozen_seed_conditions():
    plan = json.loads((staging.CFG/'pilot.json').read_text())
    assert staging.digest(staging.CFG/'pilot.json') == staging.PLAN_HASH
    images = plan['images']
    a,b = (staging.ordered(images,s) for s in staging.SEEDS)
    assert len(set(a)) == 22 and set(a) == set(b) == set(images)
    assert a != b
    assert a == staging.ordered(dict(reversed(list(images.items()))),staging.SEEDS[0])
    renamed = {str(i):images[n] for i,n in enumerate(images)}
    assert [images[n] for n in a] == [renamed[n] for n in staging.ordered(renamed,staging.SEEDS[0])]


@pytest.mark.parametrize('name',['../config.py','/config.py','models/../../pilot.json',
    'models/a.csv','conf/models.yaml.bak','tests/test_model.py','bundle/private.whl','evaluate_imc2025.py'])
def test_restricted_source(name):
    assert not staging.allowed(name,'conf/selected.yaml')


def test_copy_and_package_validation(tmp_path):
    src=tmp_path/'original'
    src.write_bytes(b'fixture')
    with pytest.raises(ValueError):
        staging.copy_checked(src,tmp_path/'bad','0'*64)
    target=tmp_path/'package/test/ETs/image_0000.png'
    staging.copy_checked(src,target,staging.digest(src))
    (tmp_path/'private.json').write_text(json.dumps({'inventory':{'test/ETs/image_0000.png':staging.digest(src)}}))
    launcher.checked_package(tmp_path)
    (target.parent/'historical.csv').write_text('forbidden')
    with pytest.raises(ValueError):
        launcher.checked_package(tmp_path)


def test_attribution_and_no_host_mounts(tmp_path):
    record=scoring.identity(20261111)
    scoring.validate_identity(record,20261111)
    with pytest.raises(ValueError):
        scoring.validate_identity(dict(record,baseline_id=record['comparison_baseline_id']),20261111)
    with pytest.raises(ValueError):
        scoring.validate_identity(record,20261112)
    with pytest.raises(ValueError):
        launcher.command('mutable:tag',tmp_path,tmp_path)
    args=launcher.command('sha256:'+'a'*64,tmp_path/'pkg',tmp_path/'out')
    assert args.count('--mount')==2 and '--network=none' in args and '--infer' not in args
