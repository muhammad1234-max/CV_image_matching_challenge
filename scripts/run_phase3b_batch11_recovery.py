"""Audited batch 11 orchestration override for the frozen Phase 3B runner."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

IMPLEMENTATION = 'd5d612c530d52089b6dfa7ab3589ccaee572d0ba6281b9bfb35ceba044e5ce93'
STAGE_G_SECONDS = 14400


def main(source, settings_path):
    source = Path(source).resolve()
    sys.path.insert(0, str(source / 'src/baselines/development'))
    import runner

    identity = runner.verify_implementation()
    if identity['implementation_hash'] != IMPLEMENTATION:
        raise ValueError('Unexpected frozen implementation')
    settings = runner.read(settings_path)
    if settings['batches'] != ['11'] or not settings['export_cache']:
        raise ValueError('Recovery requires batch 11 and cache export')
    if settings['session_remaining_seconds'] != 21600:
        raise ValueError('Start recovery with a fresh six-hour session budget')
    if len(settings['resume_archives']) != 2:
        raise ValueError('Supply the verified report and cache archives')
    work = Path(settings['work'])
    if work.exists():
        raise ValueError('Recovery work directory must be fresh')
    # Check every archive byte before trusting the existing checkpoint keys.
    import zipfile
    archives = {}
    kinds = set()
    for name in settings['resume_archives']:
        path = Path(name)
        with zipfile.ZipFile(path) as archive:
            manifest = runner.verify_zip(archive)
            if manifest['implementation_hash'] != IMPLEMENTATION or manifest['batch_id'] != '11':
                raise ValueError('Unexpected recovery archive identity')
            if any(n.startswith('datasets/') and n.split('/')[1] != 'pt_sacrecoeur_trevi_tajmahal'
                   for n in manifest['files']):
                raise ValueError('Unexpected dataset in recovery archive')
            kinds.add(manifest['kind'])
        archives[str(path)] = runner.sha(path)
    if kinds != {'report', 'cache'}:
        raise ValueError('Recovery needs one report and one cache')
    original_read = runner.read
    config_path = (runner.CONFIG / 'full_development.json').resolve()

    def recovery_read(path):
        value = original_read(path)
        if Path(path).resolve() == config_path:
            if value['sfm_stage_limit_seconds'] != 7200:
                raise ValueError('Unexpected original reconstruction limit')
            value = dict(value, sfm_stage_limit_seconds=STAGE_G_SECONDS)
        return value

    work.mkdir(parents=True)
    runner.write(work / 'recovery_orchestration.json', {
        'recovery_id': 'batch11_stage_g_4h_v1',
        'implementation_hash': IMPLEMENTATION,
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_sfm_stage_limit_seconds': 7200,
        'effective_sfm_stage_limit_seconds': STAGE_G_SECONDS,
        'session_seconds': 21600,
        'archive_reserve_seconds': 600,
        'reason': 'Prior batch 11 Stage G reached its 7200-second deadline',
        'resume_archive_sha256': archives,
        'comparison_protocol': 'Apply this allowance to batch 11 for all compared variants; record retries and their compute',
    })
    # verify_implementation uses common.read and still checks the original files.
    # Only the runner's orchestration configuration read receives the override.
    runner.read = recovery_read
    try:
        print('Recovery batch 11; Stage G limit 14400 s; session 21600 s, reserve 600 s', flush=True)
        result = runner.run(settings)
    finally:
        runner.read = original_read
    print(result['status'], flush=True)
    return 0 if result['status'] == 'BATCH_COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args()
    sys.exit(main(args.source, args.settings))
