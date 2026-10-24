"""Validate one complete refresh before publishing stable repository/Pages paths."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from .config import MARKETS
from .data import load_data
from .macro import load_macro
from .comparison import VARIANTS
from .report import write_reports


def prepare_publication(data_dir="data", report_dir="reports", destination="docs/latest"):
    data_dir, report_dir, destination = map(Path, (data_dir, report_dir, destination))
    frames, manifest = load_data(data_dir)
    macros = load_macro(data_dir, manifest)
    result = json.loads((report_dir / "analysis.json").read_text(encoding="utf-8"))
    if result["manifest"] != manifest or set(result["markets"]) != set(MARKETS):
        raise ValueError("Report and downloaded market snapshot do not match")
    previous_path = destination / "data" / "manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
    if previous and "macro" in previous and macros is None:
        raise ValueError("Refusing to remove published macro data")
    if macros is not None:
        if "macro" not in result:
            raise ValueError("Missing macro report")
        for key, meta in manifest["macro"]["sources"].items():
            prior = (previous or {}).get("macro", {}).get("sources", {}).get(key, {})
            if prior.get("last_date", "") > meta["last_date"]:
                raise ValueError(f"Refusing to replace newer published macro data: {key}")
    for code in MARKETS:
        latest = str(frames[code].index[-1].date())
        market = result["markets"][code]
        if manifest["markets"][code]["last_date"] != latest or market["summary"]["date"] != latest:
            raise ValueError(f"Inconsistent latest date for {code}")
        if {f["horizon"] for f in market["forecasts"]} != {5, 20}:
            raise ValueError(f"Missing forecast horizon for {code}")
        for forecast in market["forecasts"]:
            if forecast["origin"] != latest or forecast["last_train_label_date"] >= latest:
                raise ValueError(f"Invalid forecast timing for {code}")
        if previous and previous["markets"][code]["last_date"] > latest:
            raise ValueError(f"Refusing to replace newer published data for {code}")
        if macros is not None:
            comparisons = market.get("macro_comparisons", [])
            if {v["horizon"] for v in comparisons} != {5, 20}:
                raise ValueError(f"Missing macro comparison for {code}")
            for comparison in comparisons:
                if set(comparison["variants"]) != set(VARIANTS) or comparison["n"] < 20:
                    raise ValueError(f"Incomplete matched comparison: {code}")
                if len(comparison["records"]) != comparison["n"]:
                    raise ValueError(f"Incorrect comparison sample count: {code}")
                for record in comparison["records"]:
                    if not record["last_train_label_date"] < record["date"] < record["target_date"] or set(record["predicted_log"]) != set(VARIANTS):
                        raise ValueError(f"Invalid comparison timing or variants: {code}")
                for variant in comparison["variants"].values():
                    f = variant["forecast"]
                    if f["origin"] != latest or f["last_train_label_date"] >= latest:
                        raise ValueError(f"Invalid macro forecast timing: {code}")

    # Stable CSV names prevent one file per refresh accumulating in the published tree.
    published = copy.deepcopy(manifest)
    for code in MARKETS:
        published["markets"][code]["file"] = f"{code}.csv"
    if macros is not None:
        for key in published["macro"]["sources"]:
            published["macro"]["sources"][key]["file"] = f"macro-{key}.csv"
    result["manifest"] = published
    with tempfile.TemporaryDirectory(prefix="market-pulse-publish-") as tmp:
        stage = Path(tmp)
        data_stage = stage / "data"
        data_stage.mkdir()
        for code in MARKETS:
            shutil.copyfile(data_dir / manifest["markets"][code]["file"], data_stage / f"{code}.csv")
        if macros is not None:
            for key, meta in manifest["macro"]["sources"].items():
                shutil.copyfile(data_dir / meta["file"], data_stage / f"macro-{key}.csv")
        (data_stage / "manifest.json").write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
        load_data(data_stage)  # Verify the published CSV bytes, including all four checksums.
        load_macro(data_stage, published)
        write_reports(result, stage)
        (stage / ".nojekyll").touch()
        # Destination is touched only after every validation and render succeeds.
        shutil.copytree(stage, destination, dirs_exist_ok=True)
    rows = ["## Market dashboard updated", "", f"Downloaded at (UTC): {manifest['fetched_at_utc']}", "", "| Market | Last complete session | Rows |", "|---|---|---:|"]
    rows += [f"| {MARKETS[c]['name']} | {published['markets'][c]['last_date']} | {len(frames[c])} |" for c in MARKETS]
    rows += ["", "Updated OHLCV data, technical indicators, PCA, 5/20-session forecasts and backtests.", ""]
    if macros is not None:
        rows += ["Rates / VIX / USD-CAD / USD-TWD refreshed. Matched technical, rates, VIX, FX and combined ablations:", "", "| Market | Horizon | Technical MAE | Combined MAE | Relative improvement |", "|---|---:|---:|---:|---:|"]
        for code in ("CA", "US", "TW"):
            for comparison in result["markets"][code]["macro_comparisons"]:
                v = comparison["variants"]
                rows.append(f"| {code} | {comparison['horizon']} | {v['technical']['metrics']['mae']:.3%} | {v['all']['metrics']['mae']:.3%} | {v['all']['improvement']['relative_mae_improvement']:+.1%} |")
    summary = "\n".join(rows)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)
    print(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--destination", default="docs/latest")
    args = parser.parse_args()
    try:
        prepare_publication(args.data_dir, args.report_dir, args.destination)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Publication failed: {exc}\nThe remote dashboard was not updated.\n")


if __name__ == "__main__":
    main()
