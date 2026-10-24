import copy
import numpy as np
import pandas as pd
import pytest

from market_pulse.evaluation import (regime_labels, validate_span, paired_improvement, bootstrap_indices,
                                     holm_adjust, stability_checks)
from market_pulse.comparison import compare_models, VARIANT_GROUPS
from market_pulse.indicators import indicators
from market_pulse.macro import align_macro
from test_analysis import prices
from test_macro import macros
from test_publish import snapshot


def test_regime_threshold_uses_prior_data_and_labels_never_see_future():
    frame = indicators(prices())
    before = regime_labels(frame)
    t = 1000
    assert before.volatility_threshold.iloc[t] == pytest.approx(frame.volatility_20.iloc[t-756:t].quantile(.75))
    assert before.bull.iloc[t] == (frame.close.iloc[t] >= frame.sma200.iloc[t])
    modified = prices()
    modified.iloc[1100:, :4] *= np.linspace(1, 5, 300)[:, None]
    pd.testing.assert_frame_equal(before.iloc[:1100], regime_labels(indicators(modified)).iloc[:1100])


def test_fixed_anchor_survives_appended_future_sessions():
    def run(n):
        frame = indicators(prices(n))
        features, _ = align_macro(frame.index, 'TW', macros(n))
        return compare_models(frame.join(features), horizon=20, backtest_start='2021-01-04')
    before, after = run(1400), run(1420)
    old = {r['date']: r for r in before['records']}
    new = {r['date']: r for r in after['records']}
    assert len(new) > len(old)
    for date, row in old.items():
        assert row == new[date]
    for c in (before, after):
        assert c['slices']['bull']['n'] + c['slices']['bear']['n'] == c['n']
        assert c['slices']['high_volatility']['n'] + c['slices']['normal_volatility']['n'] == c['n']
        assert sum(y['n'] for y in c['annual'].values()) == c['n']
        assert not list(c['annual'].values())[-1]['complete']


def test_five_calendar_years_is_validated_on_actual_forecasts():
    rows = [{'date':'2020-01-02','target_date':'2025-01-01'}]
    with pytest.raises(ValueError, match='At least 5 calendar years'):
        validate_span(rows, 5)
    rows[0]['target_date']='2025-01-02'
    assert validate_span(rows, 5) >= 5


def test_full_factorial_includes_every_group_combination():
    combinations = {frozenset(g) for g in VARIANT_GROUPS.values()}
    assert len(combinations) == 8
    assert {frozenset(), frozenset(['rates','vix']), frozenset(['rates','fx']), frozenset(['vix','fx'])}.issubset(combinations)


def test_conditional_bootstrap_resamples_timeline_and_mask_jointly():
    n=80
    base=np.linspace(.02,.08,n)
    difference=np.sin(np.arange(n)/6)*.005+.001
    mask=(np.arange(n)<15)|(np.arange(n)>55)
    indices=bootstrap_indices(n, repetitions=300)
    result=paired_improvement(base, base-difference, mask=mask, indices=indices)
    expected=[]
    for sample in indices:
        selected=sample[mask[sample]]
        if len(selected): expected.append(difference[selected].mean())
    np.testing.assert_allclose([result['ci_low'],result['ci_high']],np.quantile(expected,[.025,.975]))
    assert result['mae_difference'] == pytest.approx(difference[mask].mean())


def test_small_or_empty_regimes_have_no_inference():
    base=np.linspace(.02,.08,80)
    for count in (0,1,19):
        result=paired_improvement(base, base-.001, mask=np.arange(80)<count)
        assert result['ci_low'] is result['ci_high'] is result['p_one_sided'] is None
        if count: assert result['mae_difference'] == pytest.approx(.001)
        else: assert result['relative_mae_improvement'] is None


def test_holm_known_values_and_input_order():
    assert holm_adjust([.03,.001,.01]) == pytest.approx([.03,.003,.02])
    assert holm_adjust([.9,.8]) == [1.,1.]


def stability_fixture():
    variant={'improvement':{'relative_mae_improvement':.03,'ci_low':.001,'mae_difference':.001}}
    slice_={'n':30,'sufficient':True,'variants':{'rates':variant}}
    return {'variants':{'rates':variant},'slices':{k:copy.deepcopy(slice_) for k in ['bull','bear','high_volatility','normal_volatility','early','recent']},
            'annual':{str(y):{**copy.deepcopy(slice_),'complete':y<2026} for y in range(2020,2027)}}


def test_stability_requires_evidence_every_regime_and_complete_years():
    c=stability_fixture()
    assert stability_checks(c,'rates',.01)['passes']
    assert stability_checks(c,'rates',.01)['eligible_years']==6
    assert not stability_checks(c,'rates',.2)['passes']
    c['slices']['bear']['variants']['rates']['improvement']['mae_difference']=-.001
    assert not stability_checks(c,'rates',.01)['passes']
    c=stability_fixture(); c['slices']['bear']['sufficient']=False
    assert not stability_checks(c,'rates',.01)['passes']
    c=stability_fixture()
    for year in ('2020','2021','2022'):
        c['annual'][year]['variants']['rates']['improvement']['mae_difference']=-.001
    assert not stability_checks(c,'rates',.01)['passes']


def test_publication_rejects_short_actual_coverage(snapshot):
    import json
    from market_pulse.macro import save_snapshot
    from market_pulse.publish import prepare_publication
    from market_pulse.comparison import VARIANTS
    data, reports, destination = snapshot
    manifest=json.loads((data/'manifest.json').read_text())
    sources={key:save_snapshot(data,key,frame,{}) for key,frame in macros().items()}
    manifest['macro']={'sources':sources}
    (data/'manifest.json').write_text(json.dumps(manifest))
    result=json.loads((reports/'analysis.json').read_text())
    result['manifest']=manifest; result['macro']={}
    for market in result['markets'].values():
        market['macro_comparisons']=[{'horizon':h,'n':20,'variants':{key:{} for key in VARIANTS},
            'records':[{'date':'2020-01-02','target_date':'2023-01-02'} for _ in range(20)]} for h in (5,20)]
    (reports/'analysis.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='At least 5 calendar years'):
        prepare_publication(data,reports,destination)
    assert not destination.exists()
