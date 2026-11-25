import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ns64_probe_v2 as probe


@pytest.mark.parametrize('name',['../x','/abs','a/../b','a\\b','a//b','C:/x'])
def test_reject_unsafe_archive(name):
    with pytest.raises(ValueError):
        probe.validate_entries([zipfile.ZipInfo(name)])


def test_archive_types_and_duplicates():
    link=zipfile.ZipInfo('utils/file.py')
    link.external_attr=(stat.S_IFLNK|0o777)<<16
    with pytest.raises(ValueError):
        probe.validate_entries([link])
    with pytest.raises(ValueError):
        probe.validate_entries([zipfile.ZipInfo('A'),zipfile.ZipInfo('a')])


def test_full_source_lock_and_disclosure():
    plan=json.loads((probe.staging.CFG/'pilot.json').read_text())
    lock=json.loads(probe.LOCK.read_text())
    with zipfile.ZipFile(probe.staging.CFG/'ns64_pilot_source.zip.bin') as z:
        probe.validate_entries(z.infolist())
        assert 'et_et008.png' in z.read('utils/example.py').decode()
        names=[n.encode() for n in plan['images']]
        assert 'utils/example.py' not in lock
        for n,h in lock.items():
            data=z.read(n)
            assert hashlib.sha256(data).hexdigest()==h
            assert not any(original in data for original in names)
        assert 'models/_research_only/silk/config/core.py' in lock
        assert 'models/pixloc/pixlib/utils/tools.py' in lock
        assert lock[plan['config']]==plan['config_sha256']


def test_pinned_order_conditions():
    plan=json.loads((probe.staging.CFG/'pilot.json').read_text())
    config=json.loads(probe.CONFIG.read_text())
    for seed in probe.staging.SEEDS:
        names=probe.staging.ordered(plan['images'],seed)
        digest=hashlib.sha256('\n'.join(plan['images'][n] for n in names).encode()).hexdigest()
        assert digest==config['conditions'][f'ns64_isolated_v1_ETs_{seed}']['permutation_sha256']
