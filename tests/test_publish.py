import copy
import hashlib
import json

import pandas as pd
import pytest

from market_pulse.config import MARKETS
from market_pulse.data import load_data
from market_pulse.publish import prepare_publication


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    data, reports, target = [tmp_path / p for p in ('data', 'reports', 'published')]
    data.mkdir(); reports.mkdir()
    dates = pd.bdate_range('2020-01-01', periods=820, name='date')
    frame = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 10.}, index=dates)
    csv = frame.to_csv()
    manifest = {'fetched_at_utc': '2026-10-24T17:00:00+00:00', 'markets': {}}
    result = {'markets': {}}
    for code in MARKETS:
        filename = f'{code}-download.csv'
        (data / filename).write_text(csv)
        manifest['markets'][code] = {'file': filename, 'sha256': hashlib.sha256(csv.encode()).hexdigest(), 'last_date': str(dates[-1].date())}
        result['markets'][code] = {'summary': {'date': str(dates[-1].date())}, 'forecasts': [{'horizon': h, 'origin': str(dates[-1].date()), 'last_train_label_date': str(dates[-2].date())} for h in (5, 20)]}
    result['manifest'] = copy.deepcopy(manifest)
    (data / 'manifest.json').write_text(json.dumps(manifest))
    (reports / 'analysis.json').write_text(json.dumps(result))
    def render(payload, path):
        (path / 'index.html').write_text('validated report')
        (path / 'analysis.json').write_text(json.dumps(payload))
    monkeypatch.setattr('market_pulse.publish.write_reports', render)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    return data, reports, target


def test_published_csv_is_reproducible_and_names_are_stable(snapshot):
    data, reports, target = snapshot
    prepare_publication(data, reports, target)
    frames, manifest = load_data(target / 'data')
    assert set(frames) == set(MARKETS)
    assert {v['file'] for v in manifest['markets'].values()} == {f'{c}.csv' for c in MARKETS}
    assert json.loads((target / 'analysis.json').read_text())['manifest'] == manifest
    assert (target / 'index.html').read_text() == 'validated report'


def test_failed_refresh_cannot_replace_newer_publication(snapshot):
    data, reports, target = snapshot
    prepare_publication(data, reports, target)
    previous_path = target / 'data/manifest.json'
    previous = json.loads(previous_path.read_text())
    previous['markets']['TW']['last_date'] = '2099-01-01'
    previous_path.write_text(json.dumps(previous))
    before = {str(p): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='newer published data'):
        prepare_publication(data, reports, target)
    assert before == {str(p): p.read_bytes() for p in target.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure', ['snapshot', 'missing_horizon', 'corrupt_csv'])
def test_inconsistent_or_incomplete_output_is_not_published(snapshot, failure):
    data, reports, target = snapshot
    path = reports / 'analysis.json'
    report = json.loads(path.read_text())
    if failure == 'snapshot':
        report['manifest']['markets']['US']['sha256'] = 'wrong'
    elif failure == 'missing_horizon':
        report['markets']['CA']['forecasts'].pop()
    else:
        (data / 'TW-download.csv').write_text('bad download')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        prepare_publication(data, reports, target)
    assert not target.exists()
