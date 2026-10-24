from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from .config import MARKETS
from .indicators import indicators
from .models import forecast, market_pca
from .macro import align_macro, macro_summary
from .comparison import compare_models
from .evaluation import BACKTEST_START, MIN_BACKTEST_YEARS, POLICY as EVALUATION_POLICY, add_stability_assessment


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def analyze(frames, manifest, macros=None):
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "manifest": manifest, "markets": {}, "pca": market_pca(frames)}
    if macros is not None:
        result["macro"] = macro_summary(macros, manifest)
    columns = ["open", "high", "low", "close", "volume", "sma20", "sma60", "sma200", "k", "d", "j", "rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "volatility_20", "drawdown"]
    for code, prices in frames.items():
        print(f"Analyzing {code}…", flush=True)
        frame = indicators(prices)
        latest = frame.iloc[-1]
        first_year = frame[frame.index.year < frame.index[-1].year]
        ytd = latest.close / first_year.close.iloc[-1] - 1 if not first_year.empty else None
        age = (pd.Timestamp.now(tz=MARKETS[code]["timezone"]).date() - frame.index[-1].date()).days
        summary = {"date": str(frame.index[-1].date()), "close": latest.close, "day_return": latest.return_1, "month_return": latest.return_20, "ytd": ytd, "k": latest.k, "d": latest.d, "rsi": latest.rsi, "volatility": latest.volatility_20, "drawdown": latest.drawdown, "max_drawdown": frame.drawdown.min(), "sma20_gap": latest.sma20_gap, "sma200_gap": latest.sma200_gap, "macd_hist": latest.macd_hist, "atr_pct": latest.atr_pct, "calendar_age_days": age, "stale": age > 4, "trend": "長短期均線之上" if latest.sma20_gap > 0 and latest.sma200_gap > 0 else ("長短期均線之下" if latest.sma20_gap < 0 and latest.sma200_gap < 0 else "長短期趨勢分歧")}
        history = frame.tail(1260)
        result["markets"][code] = {**MARKETS[code], "summary": summary, "forecasts": [forecast(frame, h, backtest_start=BACKTEST_START, minimum_years=MIN_BACKTEST_YEARS) for h in [5, 20]], "history": {"dates": history.index.strftime("%Y-%m-%d").tolist(), **{c: history[c].tolist() for c in columns}}}
        if macros is not None:
            aligned, audit = align_macro(frame.index, code, macros)
            comparisons = [compare_models(frame.join(aligned), h, backtest_start=BACKTEST_START, minimum_years=MIN_BACKTEST_YEARS) for h in (5, 20)]
            result["markets"][code]["macro_comparisons"] = comparisons
            used = {r["date"] for comparison in comparisons for r in comparison["records"]} | {summary["date"]}
            result["markets"][code]["macro_alignment"] = {
                key: {field: [value for date, value in zip(source["dates"], values) if date in used]
                      for field, values in source.items()} for key, source in audit.items()}
    if macros is not None:
        result["evaluation"] = {"policy": EVALUATION_POLICY, "cross_market_stable": add_stability_assessment(result["markets"])}
    return clean(result)


def text_report(result):
    lines = ["# Market Pulse 大盤研究報告", "", f"建立時間（UTC）：{result['generated_at_utc']}", "", "本報告以供應商可取得的已完成日線為準；各市場最新日期可能不同。價格指數不含股息，亦未換匯。", "", "| 市場 | 資料日期 | 收盤 | 20日報酬 | 年初至今 | K / D | RSI | 年化波動 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    pct = lambda n: f"{n:+.2%}" if n is not None else "N/A"
    for market in result["markets"].values():
        s = market["summary"]
        lines.append(f"| {market['name']} | {s['date']} | {s['close']:,.2f} | {pct(s['month_return'])} | {pct(s['ytd'])} | {s['k']:.1f} / {s['d']:.1f} | {s['rsi']:.1f} | {s['volatility']:.1%} |")
    lines += ["", "## 趨勢與預測", "", "預測是從所列資料日期起算的 5／20 個當地交易日；若資料已過期，不能當作今天的預測。80% 區間來自樣本外殘差的 10／90 分位數，並非保證覆蓋率。", ""]
    for market in result["markets"].values():
        s = market["summary"]
        lines += [f"### {market['name']}", "", f"{s['trend']}；距 200 日均線 {pct(s['sma200_gap'])}，距下載期間最高收盤價 {pct(s['drawdown'])}。K {'高於' if s['k'] > s['d'] else '低於'} D；MACD 柱狀值 {'正' if s['macd_hist'] > 0 else '負'}。"]
        if s["stale"]:
            lines += [f"資料已距今天 {s['calendar_age_days']} 個日曆日，須先更新再判讀。"]
        for f in market["forecasts"]:
            m = f["metrics"]
            verdict = "MAE 優於兩個基準" if f["beats_baselines"] else "MAE 未同時優於兩個基準，預測優勢尚未成立"
            lines += [f"- {f['horizon']} 交易日：點預測 {pct(f['return'])}；80% 經驗誤差帶 {pct(f['low'])} 至 {pct(f['high'])}。{m['n']} 次不重疊樣本外預測，MAE {m['mae']:.2%}（不變基準 {m['zero_mae']:.2%}、歷史平均 {m['mean_mae']:.2%}）；方向準確率 {m['direction_accuracy']:.1%}（永遠看漲 {m['always_up_accuracy']:.1%}）。{verdict}。"]
        lines += [""]
    if "macro" in result:
        lines += ["## 利率、VIX、匯率：共同樣本比較", "", "八組模型使用完全相同的訓練列、預測起點與目標日；標準化、PCA 與 Ridge 每次僅擬合訓練集。MAE 改善為相對技術模型的誤差下降比例，正值較好。主頁點預測仍為原技術模型，沒有按這份測試集自動挑選贏家。", "", "| 市場／期間 | 模型 | 次數 | MAE | 相對改善 | MAE 差 95% 區間（百分點） | 方向準確率 |", "|---|---|---:|---:|---:|---:|---:|"]
        for code, market in result["markets"].items():
            for comparison in market["macro_comparisons"]:
                for key, variant in comparison["variants"].items():
                    metric, improvement = variant["metrics"], variant["improvement"]
                    ci = "—" if key == "technical" else f"[{100*improvement['ci_low']:+.3f}, {100*improvement['ci_high']:+.3f}]"
                    lines.append(f"| {market['region']} {comparison['horizon']}日 | {variant['name']} | {comparison['n']} | {metric['mae']:.3%} | {pct(improvement['relative_mae_improvement'])} | {ci} | {metric['direction_accuracy']:.1%} |")
        lines += ["", "區間採配對循環區塊 bootstrap：每塊 5 次預測、4,000 次抽樣、seed=42。差值為原模型絕對誤差減新模型絕對誤差；區間跨 0 代表改善證據不足。整體改善 p 值對三個主要市場、兩個期限、七個新增組合共 42 次比較作 Holm 校正；各狀態／年度的區間未校正，只作診斷。20 日期限或空頭子群樣本仍較少。JSON 另列前／後半段誤差差值、全部預測、訓練日期雜湊及各來源對齊日期。", "", "### 總經資料與時點", "", "利率整合 Leo-CZ/fed-rates-dashboard 的固定版本歷史資料，再從 FRED 延伸與更新：2／10／30 年美債、10 年通膨損益兩平率、10／30 年實質殖利率。VIX 使用 Cboe 官方收盤；匯率使用 Yahoo CAD=X、TWD=X，分別為每 1 美元可兌換的 CAD、TWD，正報酬代表美元升值。", "", "觀測日後第 2 個日曆日 00:00 UTC 才假設可用，按各交易所當地收盤向後對齊，觀測資料超過 10 日即不用；不往回填補。這是保守延遲假設，不是真實發布時間紀錄。歷史值採最新下載版本，未重建 ALFRED 當時版本；仍可能受資料修訂影響，不能稱為完整 point-in-time 回測。", ""]
        for key, series in result["macro"]["series"].items():
            lines.append(f"- {key} 最新來源日期：{series['date']}；{series['units']}。最新來源值不一定已供當天模型使用，對齊日期見 JSON。")
        lines += ["", "[利率來源 repo](https://github.com/Leo-CZ/fed-rates-dashboard) · [FRED](https://fred.stlouisfed.org/) · [Cboe VIX 歷史資料](https://www.cboe.com/tradable_products/vix/vix_historical_data) · [ALFRED 修訂版本說明](https://alfred.stlouisfed.org/docs/alfred_capturing_data.pdf)", ""]
    if "evaluation" in result:
        lines += stability_report(result)
    p = result["pca"]
    lines += ["## 三地大盤 PCA", "", f"以 {p['start']} 至 {p['end']} 的 {p['n']} 個共同日期之標準化對數報酬分析。PC1／PC2／PC3 解釋變異比例為 " + " / ".join(f"{v:.1%}" for v in p["explained_variance"]) + "。", "", "| 主成分 | 加拿大權重 | 美國權重 | 台灣權重 |", "|---|---:|---:|---:|"]
    for i, row in enumerate(p["weights"]):
        lines.append(f"| PC{i+1} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
    lines += ["", "權重是統計主成分係數，並非成分股持股比例。PCA 只描述共變動，不證明因果。同一日期的台股比北美先收盤，相關矩陣不代表同步交易資訊；跨市場 PCA 未被用作預測特徵。", "", "## 方法與限制", "", "KD 使用台灣遞迴式 (9,3,3)，K、D 初始值皆 50；K 超過 80 是高檔動能，不能單獨推出即將下跌。RSI(14) 與 ATR(14) 採 Wilder 平滑。模型為 StandardScaler → PCA(90% 變異) → Ridge(alpha=10)，每次僅在歷史訓練集擬合，訓練標籤結束日嚴格早於預測起點。", "", "模型自 2020 年初開始評估，發布前要求實際樣本外跨度至少五年，每隔 h 日預測 h 日報酬，因此評估區間不重疊。指數成交量品質因供應商而異，不用於模型。未加入新聞、成分股權重、交易成本或可成交策略績效；本報告為研究用途，不構成買賣建議。", "", "## 來源", ""]
    for market in result["markets"].values():
        lines.append(f"- [{market['name']} · Yahoo Finance](https://finance.yahoo.com/quote/{market['symbol'].replace('^', '%5E')}/history/)")
    lines += ["- [S&P/TSX 指數介紹](https://www.spglobal.com/spdji/en/indices/equity/sp-tsx-composite-index/)", "- [臺灣證券交易所指數系列](https://www.twse.com.tw/zh/indices/indices/series.html)", "- [scikit-learn 時序切分](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)", ""]
    return "\n".join(lines)


def write_reports(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    html = template.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", data.replace("<", "\\u003c"))
    (directory / "index.html").write_text(html, encoding="utf-8")
    (directory / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    (directory / "analysis.md").write_text(text_report(result), encoding="utf-8")
    print(f"Report: {directory / 'index.html'}", flush=True)


def stability_report(result):
    lines = ["", "## 長期與市場狀態穩定性", "", "事先固定：多頭＝當時收盤≥200 日均線；空頭＝收盤<200 日均線。高波動＝自身20日年化波動≥之前756交易日的75分位數（至少252筆，門檻延後一日）。高／一般波動與多／空頭是兩種重疊分類，不可相加樣本數。這是技術狀態標籤，不是官方景氣循環或事後高低點。", "", "比較從 2020 年初開始；預測日期以固定起點每隔 h 個交易日取樣，新增一天不會重新排列歷史測試日期。每次只用當時成熟的標籤訓練。", "", "穩定候選需同時滿足：整體MAE至少改善1%、95%下界>0、42次整體比較Holm校正p<0.05；多頭、空頭、高波動、一般波動及2024年前／後各至少20次且皆改善；至少5個完整起點年度，且至少2/3年度改善。這些是研究篩選規則，不保證未來；2024年後資料曾在前次研究中檢視，不是全新保留測試集。未達標不等於指標永遠無用。", "", "| 市場 | 期限 | 樣本外期間／次數 | 整體MAE最低組合（描述性） | 通過全部穩定條件 |", "|---|---:|---|---|---|"]
    fmt = lambda value: "—" if value is None else f"{value:+.1%}"
    for code in ('CA', 'US', 'TW'):
        market = result['markets'][code]
        for c in market['macro_comparisons']:
            best = min(c['variants'], key=lambda k: c['variants'][k]['metrics']['mae'])
            passed = [v['name'] for k,v in c['variants'].items() if k != 'technical' and v['stability']['passes']]
            lines.append(f"| {market['region']} | {c['horizon']}日 | {c['test_start']}～{c['test_end']} / {c['n']} | {c['variants'][best]['name']} | {', '.join(passed) or '無'} |")
    lines += ["", "### 相對原技術模型的 MAE 改善（正值較好）", "", "區間以原始連續預測時間軸為單位，同時重抽配對誤差及狀態標籤，避免把不相鄰的空頭時期拼接成連續樣本。小於20次的子群只顯示點估計，不給區間，也不能通過穩定條件。", "", "| 市場／期限 | 組合 | 全期 | 多頭 | 空頭 | 高波動 | 一般波動 | 完整年度改善數 | Holm p |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for code in ('CA', 'US', 'TW'):
        for c in result['markets'][code]['macro_comparisons']:
            for key, v in c['variants'].items():
                if key == 'technical':
                    continue
                gains = [fmt(c['slices'][k]['variants'][key]['improvement']['relative_mae_improvement']) + f" (n={c['slices'][k]['n']})" for k in ('bull','bear','high_volatility','normal_volatility')]
                st = v['stability']
                lines.append(f"| {code} {c['horizon']}日 | {v['name']} | {fmt(v['improvement']['relative_mae_improvement'])} | " + ' | '.join(gains) + f" | {st['positive_years']}/{st['eligible_years']} | {st['holm_p']:.4f} |")
    lines += ["", "### 分年度檢查", "", "年度按預測起點歸屬，目標可跨年；未完成年度仍顯示，但不納入年度勝率。年度不足8次不納入穩定篩選；年度／狀態的CI與詳細MAE、RMSE、方向準確率保存於JSON。", ""]
    for code in ('CA', 'US', 'TW'):
        for c in result['markets'][code]['macro_comparisons']:
            years = list(c['annual'])
            lines += [f"#### {code} · {c['horizon']}日", "", '| 組合 | ' + ' | '.join(y + ('' if c['annual'][y]['complete'] else '（未完整）') for y in years) + ' |', '|---|' + '---:|'*len(years)]
            for key, v in c['variants'].items():
                if key != 'technical':
                    lines.append('| ' + v['name'] + ' | ' + ' | '.join(fmt(c['annual'][y]['variants'][key]['improvement']['relative_mae_improvement']) for y in years) + ' |')
            lines += ['']
    lines += ["[時序驗證方法](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)；[NBER 景氣日期採事後判定](https://www.nber.org/research/business-cycle-dating)，因此本模型使用當時已知價格作狀態分類。", ""]
    return lines
