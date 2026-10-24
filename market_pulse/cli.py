import argparse
from .data import load_data, refresh
from .report import analyze, write_reports
from .macro import load_macro


def main():
    parser = argparse.ArgumentParser(description="Canada / US / Taiwan market research")
    parser.add_argument("command", choices=["refresh", "build"], help="refresh downloads real data; build uses verified cached data")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--as-of", help="Optional last historical date YYYY-MM-DD; today's incomplete session always excluded")
    args = parser.parse_args()
    if args.command == "build" and args.as_of:
        parser.error("--as-of applies only to refresh; cached data is never silently truncated")
    try:
        frames, manifest = refresh(args.data_dir, args.start, args.as_of) if args.command == "refresh" else load_data(args.data_dir)
        write_reports(analyze(frames, manifest, load_macro(args.data_dir, manifest)), args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Market Pulse failed: {exc}\nNo synthetic market data was substituted.\n")


if __name__ == "__main__":
    main()
