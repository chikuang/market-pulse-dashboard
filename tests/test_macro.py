import copy
import json
import numpy as np
import pandas as pd
import pytest

from market_pulse.config import MARKETS, FEATURES
from market_pulse.macro import GROUPS, RATE_COLUMNS, align_macro, source_features, parse_rates, parse_fx, download_fred, save_snapshot, load_macro
from market_pulse.comparison import compare_models, paired_improvement, VARIANTS
from market_pulse.indicators import indicators
from market_pulse.publish import prepare_publication
from test_analysis import prices
from test_publish import snapshot


def macros(n=1400):
    dates = prices(n).index
    u = np.arange(n)
    rates = pd.DataFrame({c: 2+i*.2+np.sin(u/(25+i))*.3 for i, c in enumerate(RATE_COLUMNS)}, index=dates)
    return {"rates": rates, "rates_seed": rates.copy(),
            "vix": pd.DataFrame({"close": 20+np.sin(u/8)*5}, index=dates),
            "usdcad": pd.DataFrame({"close": 1.3+np.sin(u/40)*.1}, index=dates),
            "usdtwd": pd.DataFrame({"close": 30+np.sin(u/30)*2}, index=dates)}


def test_parse_original_repo_and_fred_missing_values():
    seed = 'date,DGS2,DGS10,DGS30,BE10,DFII10,DFII30\n2020-01-01,1,2,3,1.5,.5,1.5\n'
    out = parse_rates(seed, seed=True)
    assert list(out) == RATE_COLUMNS
    assert out.T10YIE.iloc[0] == 1.5
    fred = parse_rates('observation_date,DGS2\n2020-01-01,.\n2020-01-02,1.5\n')
    assert pd.isna(fred.iloc[0, 0])
    assert fred.iloc[1, 0] == 1.5


def test_fx_uses_only_closed_utc_dates_and_validates_symbol():
    dates = pd.date_range('2026-10-20', periods=4, tz='UTC')
    payload = {'chart': {'result': [{'meta': {'symbol': 'CAD=X'}, 'timestamp': [int(d.timestamp()) for d in dates],
        'indicators': {'quote': [{'close': [1.3, None, 1.4, 9.9], 'high': [0, 0, 0, 0]}]}}]}}
    frame = parse_fx(payload, 'CAD=X', pd.Timestamp('2026-10-23'))
    assert frame.close.tolist() == [1.3, 1.4]
    assert frame.attrs['missing_close_rows_dropped'] == 1
    with pytest.raises(ValueError, match='Unexpected FX'):
        parse_fx(payload, 'TWD=X', pd.Timestamp('2026-10-23'))


def test_fred_timeout_uses_same_series_and_records_actual_url(monkeypatch):
    def get(url, attempts):
        if '&coed=' in url:
            raise RuntimeError('timeout')
        return 'observation_date,DGS10\n2026-10-22,4.96\n'
    monkeypatch.setattr('market_pulse.macro.get_text', get)
    frame, url = download_fred('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2016-01-01&coed=2026-10-23')
    assert '&coed=' not in url and 'id=DGS10' in url
    assert frame.DGS10.iloc[0] == 4.96


@pytest.mark.parametrize('code', ['CA', 'US', 'TW'])
def test_backward_alignment_never_uses_unavailable_observation(code):
    data = macros()
    features, audit = align_macro(data['rates'].index, code, data)
    h, m = map(int, MARKETS[code]['close'].split(':'))
    origins = (features.index+pd.Timedelta(hours=h, minutes=m)).tz_localize(MARKETS[code]['timezone']).tz_convert('UTC')
    for record in audit.values():
        available = pd.to_datetime(record['available_at_utc'], utc=True)
        observed = pd.to_datetime(record['observation_dates'], utc=True)
        valid = available.notna()
        assert (available[valid] <= origins[valid]).all()
        assert ((available[valid]-observed[valid]) == pd.Timedelta(days=2)).all()
    # Monday's features can use Friday, but never Monday's North American close in Taiwan.
    monday = next(i for i, d in enumerate(features.index) if i > 50 and d.weekday() == 0)
    assert pd.Timestamp(audit['vix']['observation_dates'][monday]).weekday() == 4


def test_stale_sources_are_not_filled_forever():
    data = macros()
    data['vix'] = data['vix'].iloc[:-30]
    features, _ = align_macro(data['rates'].index, 'TW', data)
    assert features.vix_level.iloc[-1:].isna().all()
    assert features.yield10.iloc[-1:].notna().all()


def test_source_features_do_not_see_future_and_fx_direction():
    data = macros()
    before, _ = align_macro(data['rates'].index, 'TW', data)
    for frame in data.values():
        frame.iloc[-100:] *= 3
    after, _ = align_macro(data['rates'].index, 'TW', data)
    pd.testing.assert_frame_equal(before.iloc[:-100], after.iloc[:-100])
    fx = data['usdcad'].close
    assert source_features(data)['usdcad'].usdcad_return5.iloc[100] == pytest.approx(np.log(fx.iloc[100]/fx.iloc[95]))


def test_comparison_preserves_targets_and_purges_common_training_rows():
    frame = indicators(prices())
    features, _ = align_macro(frame.index, 'US', macros())
    frame = frame.join(features)
    frame.loc[frame.index[-80:-70], 'yield10'] = np.nan
    out = compare_models(frame, horizon=20)
    assert set(out['variants']) == set(VARIANTS)
    common = frame[FEATURES + sum(GROUPS.values(), [])].notna().all(axis=1).to_numpy()
    for row in out['records']:
        i = frame.index.get_loc(row['date'])
        assert row['target_date'] == str(frame.index[i+20].date())
        assert row['actual_log'] == pytest.approx(np.log(frame.close.iloc[i+20]/frame.close.iloc[i]))
        assert row['train_n'] == common[:i-20].sum()
        assert row['last_train_label_date'] < row['date']
        assert set(row['predicted_log']) == set(VARIANTS)
    for a, b in zip(out['records'], out['records'][1:]):
        assert a['target_date'] <= b['date']


def test_future_macro_change_cannot_change_past_forecast():
    frame = indicators(prices())
    data = macros()
    f, _ = align_macro(frame.index, 'TW', data)
    before = compare_models(frame.join(f), horizon=20)
    for series in data.values():
        series.iloc[-100:] *= 2
    f, _ = align_macro(frame.index, 'TW', data)
    after = compare_models(frame.join(f), horizon=20)
    cutoff = str(frame.index[-100].date())
    for a, b in zip(before['records'], after['records']):
        if a['date'] < cutoff:
            for key in VARIANTS:
                assert a['predicted_log'][key] == pytest.approx(b['predicted_log'][key], abs=1e-12)


def test_latest_missing_macro_refuses_partial_comparison():
    frame = indicators(prices())
    f, _ = align_macro(frame.index, 'US', macros())
    f.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match='Latest macro'):
        compare_models(frame.join(f), horizon=20)


def test_bootstrap_pairing_sign_and_reproducibility():
    base = np.linspace(.01, .05, 100)
    same = paired_improvement(base, base)
    assert same['ci_low'] == same['ci_high'] == 0
    out = paired_improvement(base, base-.005)
    assert out == paired_improvement(base, base-.005)
    assert out['ci_low'] == pytest.approx(.005)
    assert out['ci_high'] == pytest.approx(.005)


def test_macro_cache_checksums_and_publication_are_atomic(snapshot):
    data, reports, target = snapshot
    prepare_publication(data, reports, target)
    before = {str(p): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    sources = {key: save_snapshot(data, key, frame, {}) for key, frame in macros().items()}
    manifest = json.loads((data/'manifest.json').read_text())
    manifest['macro'] = {'sources': sources}
    assert set(load_macro(data, manifest)) == set(sources)
    (data/'manifest.json').write_text(json.dumps(manifest))
    report = json.loads((reports/'analysis.json').read_text())
    report['manifest'] = copy.deepcopy(manifest)
    report['macro'] = {}
    (reports/'analysis.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Missing macro comparison'):
        prepare_publication(data, reports, target)
    assert before == {str(p): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    (data/sources['vix']['file']).write_text('corrupt')
    with pytest.raises(ValueError, match='Macro checksum'):
        load_macro(data, manifest)
