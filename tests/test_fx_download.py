import copy
import json

import pandas as pd
import pytest

from market_pulse.macro import download_fx


def fx_payload():
    dates = pd.bdate_range('2020-01-01', periods=805, tz='UTC')
    return {'chart': {'result': [{
        'meta': {'symbol': 'CAD=X'},
        'timestamp': [int(d.timestamp()) for d in dates],
        'indicators': {'quote': [{'close': [1.3] * 805}]},
    }], 'error': None}}


@pytest.mark.parametrize('failure', [
    'empty', 'wrong_symbol', 'invalid_json', 'missing_quotes', 'invalid_price',
    'short_history', 'http_error',
])
def test_fx_recovers_from_invalid_response_without_relaxing_validation(monkeypatch, failure):
    good = fx_payload()
    bad = copy.deepcopy(good)
    item = bad['chart']['result'][0]
    if failure == 'empty':
        bad['chart'] = {'result': None, 'error': {'description': 'Temporarily unavailable'}}
    elif failure == 'wrong_symbol':
        item['meta']['symbol'] = 'TWD=X'
    elif failure == 'missing_quotes':
        item['indicators']['quote'] = []
    elif failure == 'invalid_price':
        item['indicators']['quote'][0]['close'][0] = -1
    elif failure == 'short_history':
        item['timestamp'] = item['timestamp'][:10]
        item['indicators']['quote'][0]['close'] = [1.3] * 10
    responses = [json.dumps(bad), json.dumps(good)]
    if failure == 'invalid_json':
        responses[0] = '<html>Service unavailable</html>'
    if failure == 'http_error':
        responses[0] = RuntimeError('Macro download failed: HTTP 503')
    calls, sleeps = [], []

    def get(url, **kwargs):
        calls.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr('market_pulse.macro.get_text', get)
    monkeypatch.setattr('market_pulse.macro.time.sleep', sleeps.append)
    frame = download_fx('https://example.test/CAD=X', 'CAD=X', pd.Timestamp('2024-01-01'))
    assert len(frame) == 805
    assert frame.close.eq(1.3).all()
    assert frame.attrs['missing_close_rows_dropped'] == 0
    assert len(calls) == 2
    assert sleeps == [1]


def test_fx_persistent_wrong_currency_stops_after_three_attempts(monkeypatch):
    payload = fx_payload()
    payload['chart']['result'][0]['meta']['symbol'] = 'TWD=X'
    calls, sleeps = [], []

    def get(url, **kwargs):
        calls.append(url)
        return json.dumps(payload)

    monkeypatch.setattr('market_pulse.macro.get_text', get)
    monkeypatch.setattr('market_pulse.macro.time.sleep', sleeps.append)
    with pytest.raises(RuntimeError, match='CAD=X after 3 attempts.*received.*TWD=X'):
        download_fx('https://example.test/CAD=X', 'CAD=X', pd.Timestamp('2024-01-01'))
    assert len(calls) == 3
    assert sleeps == [1, 2]
