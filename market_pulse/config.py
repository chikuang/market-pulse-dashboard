MARKETS = {
    "CA": {"symbol": "^GSPTSE", "name": "S&P/TSX 綜合指數", "region": "加拿大", "currency": "CAD", "timezone": "America/Toronto", "close": "16:00", "color": "#53d3ad"},
    "US": {"symbol": "^GSPC", "name": "S&P 500", "region": "美國", "currency": "USD", "timezone": "America/New_York", "close": "16:00", "color": "#82aaff"},
    "TW": {"symbol": "^TWII", "name": "台灣加權指數", "region": "台灣", "currency": "TWD", "timezone": "Asia/Taipei", "close": "13:30", "color": "#ffb66e"},
    "NASDAQ": {"symbol": "^IXIC", "name": "Nasdaq 綜合指數", "region": "美國・比較", "currency": "USD", "timezone": "America/New_York", "close": "16:00", "color": "#bd9bff"},
}
PRIMARY = ["CA", "US", "TW"]
FEATURES = ["return_1", "return_5", "return_20", "sma20_gap", "sma60_gap", "sma200_gap", "k", "d", "rsi", "macd_pct", "macd_hist_pct", "bb_position", "volatility_20", "atr_pct", "drawdown"]
