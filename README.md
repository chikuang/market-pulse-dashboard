# Market Pulse Dashboard

以 **加拿大、美國、台灣大盤**為核心的 Python 研究 repo。下載真實日線，計算技術指標、主成分分析（PCA），以按時間推進的回測檢驗 5／20 個交易日預測，產生可離線開啟的繁體中文互動報告。

[線上儀表板](https://chikuang.github.io/market-pulse-dashboard/) · [最新分析](docs/latest/analysis.md) · [每日更新紀錄](https://github.com/chikuang/market-pulse-dashboard/actions/workflows/report.yml)

## 快速開始

需要 Python 3.11+。在 repo 根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m market_pulse.cli refresh
```

Windows 使用 `.venv\Scripts\activate`。完成後直接以瀏覽器開啟 `reports/index.html`；Plotly 與資料皆內嵌，不依賴 CDN。也會輸出 `reports/analysis.md` 和 `reports/analysis.json`。

```bash
# 使用 SHA-256 驗證過的本地資料重建，不連線
python -m market_pulse.cli build
# 歷史截止日（按各市場當地日期，含該日；仍排除執行當天）
python -m market_pulse.cli refresh --start 2016-01-01 --as-of 2026-10-23
# 測試不需下載行情
python -m pytest -q
```

首次離線儀表板：下載 [market-dashboard-2026-10-24.html](docs/market-dashboard-2026-10-24.html) 後用瀏覽器開啟（GitHub 檔案頁不執行 HTML）。[資料來源紀錄](docs/data-manifest-2026-10-24.json) 保存首次分析使用的日期與雜湊。

完整模型需約五年以上的資料；預設由 2016 年起下載。市場資料若不足，會明確報錯。`--data-dir` 與 `--output-dir` 可更改輸出位置。

## 市場與功能

| 市場 | 指數 | Yahoo 代號 | 定位 |
|---|---|---|---|
| 加拿大 | S&P/TSX Composite | `^GSPTSE` | 加拿大核心大盤 |
| 美國 | S&P 500 | `^GSPC` | 美國大型股大盤 |
| 台灣 | TAIEX 加權指數 | `^TWII` | 台灣上市大盤 |
| 美國 | Nasdaq Composite | `^IXIC` | 比較序列，不重複納入三地 PCA |

- 日 K 線、20／60／200 日均線、20 日布林通道、同起點 100 的跨市場比較。
- 台灣遞迴式 KD(9,3,3)、J、RSI(14)、MACD(12,26,9)、ATR(14)、20 日年化波動、下載期間回撤。J 和成交量保留於 JSON；指數成交量不作模型特徵。
- 三地報酬相關矩陣、PCA 解釋變異、特徵向量權重、相關負荷與 scores。
- 5／20 個當地有效日線後的報酬／指數點預測、80% 經驗誤差帶、時間外回測。
- 6 個月／1／3／5 年視窗、市場切換、縮放、圖例切換與 JSON 匯出。
- [首次實際分析](docs/analysis-2026-10-24.md)，資料截止日依各市場標示。

「主成分」在本 repo 指統計 PCA；**未取得即時成分股／產業權重**，不能將 PCA 權重當作台積電或金融、能源類股對指數的貢獻。

## 指標定義

- `RSV = 100 × (close - low_9) / (high_9 - low_9)`。
- `K_t = 2/3 K_(t-1) + 1/3 RSV_t`，`D_t = 2/3 D_(t-1) + 1/3 K_t`。前 8 日為 NA，先前 K=D=50；9 日價格區間完全平坦時保留前 K、D。`J=3K-2D`。
- 此 KD 為台灣常用遞迴平滑，與一些平台的 SMA 型 slow stochastic 不同。K>80／<20 分別代表近期收盤接近區間高／低檔，不能單獨視為轉折。
- RSI 與 ATR 使用 Wilder 平滑，以首個完整期間的均值初始化；持續上漲 RSI=100、持續下跌=0、完全平盤=50。
- MACD 是 EMA12−EMA26，訊號線 EMA9；柱狀值是 MACD−訊號線（未乘二）。布林標準差採 `ddof=0`。
- 年化波動為 20 日對數報酬的樣本標準差 × √252；回撤以下載資料期間最高收盤價計算。

## PCA 與預測方法

**描述性 PCA**：各市場先算日 log 報酬，再取共同日期，使用最近最多 756 筆共同日期。StandardScaler 將三地報酬等尺度化，再做 PCA。符號以該 PC 絕對值最大係數為正作固定方向，符號本身無經濟意義。同日台灣先於北美收盤，且假日使報酬涵蓋期間不一致，這不是同步因果分析。

**預測用 PCA**：每個市場獨立，以自身當時已知的報酬、均線乖離、KD、RSI、MACD、布林位置、波動率、ATR、回撤共 15 項特徵建立 `StandardScaler → PCA(90% variance) → Ridge(alpha=10)`。目標是 `log(close[t+h]/close[t])`。原技術模型不使用跨市場 PCA 或新聞。新增總經比較沿用相同演算法，僅改變特徵；跨市場資訊按下述保守時間政策對齊。

**時間切分**：至少 504 個成熟訓練樣本；擴張訓練窗，約最近 504 個交易日做測試。每隔 h 日預測 h 日，因此測試報酬期間不重疊。每次重新擬合 scaler、PCA、模型；訓練標籤結束日期必須嚴格早於測試起點。模型設定固定，未用測試資料選參數。

**驗證指標**：簡單報酬的 MAE、RMSE、方向準確率，與 0% 報酬、訓練集平均 log 報酬、永遠看漲基準比較。MAE 略低不代表統計顯著，也不是扣除交易成本的策略報酬。

**區間**：最後點預測加上樣本外 log 報酬殘差的第 10／90 分位數，再轉回簡單報酬，作為 80% 經驗誤差帶。它是歷史殘差校準，不保證未來涵蓋率。另報告僅使用前 20 筆以上已知誤差所估區間的 prequential 覆蓋率。20 日期限僅約 26 次測試、6 次覆蓋率評估，推論能力有限。

## 利率、VIX、匯率整合與回測

每天除了股價，也下載下列資料，不需要 API key：

| 資料 | 來源與方向 | 固定模型特徵 |
|---|---|---|
| 利率 | [fed-rates-dashboard](https://github.com/Leo-CZ/fed-rates-dashboard) 的 `data/fred/rates_daily.csv` + FRED | 2／10／30 年美債、10−2 年及 30−10 年利差、10 年通膨損益兩平率、10／30 年實質殖利率、10 年殖利率 5 觀測日變化，共 9 項 |
| VIX | [Cboe 官方日線](https://www.cboe.com/tradable_products/vix/vix_historical_data) | 收盤、5 觀測日點數變化、相對 20 日均值，共 3 項 |
| USD/CAD、USD/TWD | Yahoo `CAD=X`、`TWD=X`，分別為 CAD／1 USD、TWD／1 USD | 各自 5／20 觀測日 log 變化，共 4 項；正值代表美元升值 |

利率來源 repo 固定於 commit `def6a4d5a9a75e1a123476750a80b9aa5c8576da`，作為可稽核歷史底稿；保留六個原始序列，`BE10` 對應 FRED `T10YIE`。每次從 FRED 重新下載 `DGS2,DGS10,DGS30,T10YIE,DFII10,DFII30` 以延伸至最新並修訂重疊值，FRED 有值時優先；原始 CSV 的 SHA-256、重疊修訂個數及底稿獨有值個數寫入 manifest。沒有執行來源 repo 的程式。利率數值單位是年百分率，利差／變化是百分點。FedWatch 快照沒有足夠長的連續歷史，因此沒有輸入回測。

**可用時間假設**：每筆宏觀觀測在觀測日後第 2 個日曆日 00:00 UTC 才可用。依加拿大／美國 16:00、台灣 13:30 當地收盤（含夏令時間）向後 `asof` 對齊，不往回補值、不插值。超過 10 日的觀測不使用。先在各來源的有效觀測序列上計算變化，再對齊股票交易日；假日不製造零報酬。圖表展示最新來源值，模型可能使用較早的值，兩者的日期分開呈現。

**重要限制**：這是保守的延遲假設，不是實際發布時間紀錄。FRED／Yahoo 歷史值使用最新下載版本，未逐日重建 [ALFRED 當時可見版本](https://alfred.stlouisfed.org/docs/alfred_capturing_data.pdf)，資料修訂仍可能影響回測。結果應稱為「延遲對齊、最新歷史版本的回溯比較」，不能當成完整 point-in-time 或可執行交易策略回測。

**五組消融比較**：原技術、技術＋利率、技術＋VIX、技術＋匯率、技術＋全部（15／24／18／19／31 項特徵）。每個市場、每個期限的五組模型，使用全部特徵皆有效的**同一批訓練列、同一預測起點與同一目標日**。資料缺口不壓縮股價日期索引，5／20 日目標仍按原有效日線計數。至少 504 個成熟訓練列，訓練標籤截止嚴格早於預測起點；scaler、PCA、Ridge 各自只擬合訓練列。共同樣本技術基準可能與主頁原始技術模型使用不同訓練歷史，因此只在這五組內比較改善。

評估最近約 504 個交易日，以 h 日間隔取得不重疊樣本外報酬；列出 MAE、RMSE、方向準確率、不變與訓練均值基準、永遠看漲準確率。正的「相對 MAE 改善」代表誤差下降。誤差差值採**配對循環移動區塊 bootstrap**（每塊 5 次預測，2,000 次抽樣，seed=42）估計 95% 區間，另列前半／後半段平均差值。區間跨 0 時不足以確認改善；未作多重比較修正，20 日約 26 次評估，統計推論有限。

這些是固定設定的探索性比較；未按測試集選擇模型或參數，**預設點預測保留原技術模型**，新增各模型最新預測放在比較表。結果不一定改善，不宣稱因果或扣除交易成本後獲利。JSON 保存全部共同預測、訓練日期雜湊、樣本數、來源對齊日期與假設；原始 CSV 可離線重現：

```bash
python -m market_pulse.cli build --data-dir docs/latest/data --output-dir reports/reproduced
```

## 資料管理

來源為 Yahoo Finance 公開但非官方 chart endpoint；無需 API key，可能限流或變更。索引幣別分別為 CAD／USD／TWD，但指數本身以點數顯示。這些是**價格指數，不含股息、換匯、費用**。

只使用當地日曆日期早於執行當天的 OHLC 日線，寧可延後納入，避免混入盤中資料。各市場不補假日、不插值。OHLC 任一缺漏的資料列會移除並記錄數量；有效日線計數可能因資料缺漏而跨過多於預期的實際交易日。提供者的重複日期、無效價格／區間會報錯。距報告產生日超過 4 日曆日顯示過期提醒（不是正式交易日曆判斷）。

`data/manifest.json` 記錄下載時間、來源 URL、各地截止時間、首末日、筆數、移除的 OHLC 缺列、CSV SHA-256。CSV 以內容雜湊命名；所有市場與總經來源成功後才原子更新 manifest，避免失敗下載破壞舊快取。`build` 驗證雜湊。本機 `data/`、`reports/` 不進 Git；每日成功發布的最新 OHLCV、manifest、JSON、Markdown 與離線 HTML 保存在 `docs/latest/`，歷史版本由 Git 留存。Yahoo 的使用與再散布規則另行適用，請依需求確認資料授權。

## GitHub Actions

- **Test analysis**：push／PR 執行測試，Python 3.11 與 3.13。
- **Update market dashboard**：每天美東時間 **00:17**（`America/New_York`，自動處理夏令時間）執行；也可在 Actions → Run workflow 手動更新。主要程式／workflow 變更時會重建。
- 每次下載加拿大、美國、台灣與 Nasdaq 的完整歷史日線、利率、VIX、USD/CAD、USD/TWD，重算所有指標、PCA、5／20 日預測與五組共同樣本回測，驗證後提交到 `main` 的 `docs/latest/`，並部署到 [GitHub Pages](https://chikuang.github.io/market-pulse-dashboard/)。使用 Actions 內建 token，無需個人 token 或 API key。
- 最新報告：`docs/latest/index.html`、`analysis.md`、`analysis.json`；可重現資料：`docs/latest/data/{CA,US,TW,NASDAQ}.csv` 、`macro-{rates,rates_seed,vix,usdcad,usdtwd}.csv` 和 `manifest.json`。每次 run 另保存 30 天的完整下載 artifact。
- 排程換算：台北時間夏令期 12:17、冬令期 13:17。使用各市場當地日期早於執行日的完整日線；周末、休市日仍檢查，市場資料日期可能不變。各指數自己的截止日會顯示在儀表板。
- 先通過測試及所有市場的資料／報告一致性驗證，才寫入最新版本；下載失敗、雜湊不符、缺少預測或資料日期倒退會讓 workflow 失敗，保留上一份成功的遠端儀表板。發布期間若 main 被其他提交更新，停止並要求重新執行，避免把舊程式生成的報告覆蓋到新版本。
- 同一時間只執行一個更新／部署；自動提交使用 `GITHUB_TOKEN`，不會遞迴觸發 workflow。GitHub 排程可能延遲，並非精準到分鐘的保證；執行紀錄及失敗原因見 Actions。
- 電腦上的舊 `file://…/reports/index.html` 是靜態快照，不會被雲端排程同步；請使用線上儀表板網址查看每日更新。

本機重建並準備相同的發布目錄：

```bash
python -m market_pulse.cli refresh
python -m market_pulse.publish
```

工作流程依據：[GitHub 排程與時區](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)、[GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 專案結構

```text
market_pulse/
  config.py        市場與特徵
  data.py          下載、品質檢查、快取
  indicators.py    技術指標
  models.py        PCA、成熟標籤回測
  macro.py         利率／VIX／匯率下載與延後對齊
  comparison.py    五組共同樣本回測與配對區塊 bootstrap
  report.py        JSON／Markdown／離線 HTML 報告
  dashboard.html   互動報告模板
  cli.py           命令列
  publish.py       發布前一致性與日期驗證
 tests/            指標、時間切分、未來資料與資料品質測試
 docs/             首次實際分析與方法來源
```

## 參考

- [利率資料整合與原始功能參考：fed-rates-dashboard](https://github.com/Leo-CZ/fed-rates-dashboard)。本 repo 另行實作，沒有複製原 repo 程式碼。
- [S&P/TSX Composite](https://www.spglobal.com/spdji/en/indices/equity/sp-tsx-composite-index/)、[TWSE 指數系列](https://www.twse.com.tw/zh/indices/indices/series.html)。
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)、[時序切分](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。本 repo 自行實作逐期成熟標籤切分。
- [Fidelity 隨機指標概念](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fast-stochastic)；本 repo 的 KD 遞迴公式以本文件定義為準。

研究用途，不構成個人化投資建議。資料、模型與歷史關係都可能失效。
