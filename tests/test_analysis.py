import numpy as np
import pandas as pd
import pytest
from market_pulse.data import validate_ohlcv, parse_chart
from market_pulse.config import MARKETS, FEATURES
from market_pulse.indicators import indicators, wilder
from market_pulse.models import forecast, training_positions, market_pca, pipeline


def prices(n=1400, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, n)))
    return pd.DataFrame({'open': close*.999, 'high': close*1.01, 'low': close*.99, 'close': close, 'volume': 1000}, index=pd.bdate_range('2018-01-01', periods=n, name='date'))


def test_kd_hand_calculation_and_bounds():
    x = prices(40)
    x[['open', 'high', 'low', 'close']] = [10., 12., 8., 11.]
    out = indicators(x)
    assert out.k.iloc[:8].isna().all()
    assert out.k.iloc[8] == pytest.approx(2*50/3 + 75/3)
    assert out.d.iloc[8] == pytest.approx(2*50/3 + out.k.iloc[8]/3)
    assert out.k.dropna().between(0, 100).all()
    assert out.d.dropna().between(0, 100).all()


def test_flat_prices_and_wilder_initialization():
    x = prices(230)
    x[['open', 'high', 'low', 'close']] = 100.
    out = indicators(x)
    assert out.k.iloc[-1] == out.d.iloc[-1] == out.rsi.iloc[-1] == 50
    assert out.bb_position.iloc[-1] == .5
    assert out.atr.iloc[-1] == out.volatility_20.iloc[-1] == 0
    result = wilder(pd.Series([1., 2., 3., 4.]), 3)
    assert result.iloc[2] == 2
    assert result.iloc[3] == pytest.approx(8/3)


def test_rsi_monotonic_price_boundaries():
    x = prices(80)
    x.close = np.arange(1., 81.)
    assert indicators(x).rsi.iloc[-1] == 100
    x.close = np.arange(80., 0., -1)
    assert indicators(x).rsi.iloc[-1] == 0


def test_indicators_do_not_see_future():
    x = prices(500)
    before = indicators(x).iloc[:350]
    x.iloc[350:, :4] *= 10
    pd.testing.assert_frame_equal(before, indicators(x).iloc[:350])


def test_purge_and_nonoverlapping_forecasts():
    f = forecast(indicators(prices()), horizon=20)
    assert f['metrics']['n'] >= 20
    assert f['low'] < f['high']
    assert f['last_train_label_date'] < f['origin']
    for row in f['records']:
        assert row['last_train_label_date'] < row['date'] < row['target_date']
    for left, right in zip(f['records'], f['records'][1:]):
        assert left['target_date'] <= right['date']
    idx = training_positions(1000, 20)
    assert idx[-1] + 20 < 1000


def test_future_changes_leave_earlier_predictions_unchanged():
    x = prices()
    first = forecast(indicators(x), 20)
    x.iloc[-120:, :4] *= np.linspace(1, 1.5, 120)[:, None]
    second = forecast(indicators(x), 20)
    cutoff = str(x.index[-120].date())
    for a, b in zip(first['records'], second['records']):
        if a['target_date'] < cutoff:
            assert a['predicted_log'] == pytest.approx(b['predicted_log'], abs=1e-12)
            assert a['actual_log'] == pytest.approx(b['actual_log'], abs=1e-12)


def test_scaler_fit_only_on_training_window():
    x = indicators(prices()).dropna(subset=FEATURES)[FEATURES]
    model = pipeline().fit(x.iloc[:550], np.arange(550))
    np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, x.iloc[:550].mean())


def test_pca_variance_orthogonality_and_holidays():
    frames = {key: prices(seed=i) for i, key in enumerate(['CA', 'US', 'TW'])}
    frames['TW'] = frames['TW'].drop(frames['TW'].index[::11])
    p = market_pca(frames)
    assert sum(p['explained_variance']) == pytest.approx(1)
    np.testing.assert_allclose(np.array(p['weights']) @ np.array(p['weights']).T, np.eye(3), atol=1e-12)
    assert all(pd.Timestamp(d) in frames['TW'].index for d in p['dates'])


def test_validation_rejects_invalid_prices():
    x = prices()
    x.iloc[3, x.columns.get_loc('high')] = 1
    with pytest.raises(ValueError, match='OHLC range'):
        validate_ohlcv(x)
    with pytest.raises(ValueError, match='unique'):
        validate_ohlcv(pd.concat([prices(), prices()]))


def test_downloader_drops_incomplete_local_day():
    dates = pd.bdate_range('2020-01-01', periods=900, tz='UTC') + pd.Timedelta(hours=14)
    q = {'open': [100.]*900, 'high': [101.]*900, 'low': [99.]*900, 'close': [100.]*900, 'volume': [0]*900}
    payload = {'chart': {'result': [{'meta': {'symbol': '^GSPC'}, 'timestamp': [int(d.timestamp()) for d in dates], 'indicators': {'quote': [q]}}]}}
    out = parse_chart(payload, MARKETS['US'], dates[-1].to_pydatetime())
    assert len(out) == 899
    assert out.index[-1].date() < dates[-1].date()


def test_internal_model_gap_is_an_error():
    x = indicators(prices())
    x.iloc[-10, x.columns.get_loc('rsi')] = np.nan
    with pytest.raises(ValueError, match='Missing technical features'):
        forecast(x)


def test_historical_as_of_is_inclusive_in_local_calendar():
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from market_pulse.data import market_cutoff
    now = datetime(2026, 10, 24, 17, tzinfo=timezone.utc)
    for spec in MARKETS.values():
        cutoff = market_cutoff(now, spec, '2026-09-30')
        local = cutoff.astimezone(ZoneInfo(spec['timezone']))
        assert str(local.date()) == '2026-10-01'
        assert local.hour == 0
        assert market_cutoff(now, spec, '2030-01-01') == now
