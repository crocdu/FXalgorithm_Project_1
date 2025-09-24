# Analyse_8.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def monthly_weighted_buy_price(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    # Your dates look like 27/4/2016
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])

    need = {"Date", "Close", "day_units"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing column(s): {miss}")

    # Only actual buy days
    d = df.loc[df["day_units"] > 0, ["Date", "Close", "day_units"]].copy()
    if d.empty:
        raise ValueError("No rows with day_units > 0 were found.")

    d["Month"] = d["Date"].dt.to_period("M")
    d["weighted_close"] = d["Close"] * d["day_units"]

    # Monthly weighted average on buy days
    out = (
        d.groupby("Month", as_index=False)
         .agg(
             units_bought=("day_units", "sum"),
             sum_weighted_close=("weighted_close", "sum"),
             buy_days=("day_units", "size"),
         )
    )
    out["weighted_avg_close"] = out["sum_weighted_close"] / out["units_bought"]
    out["MonthStart"] = out["Month"].dt.to_timestamp()
    out = out.sort_values("MonthStart").reset_index(drop=True)

    # ---- Fill missing calendar months with NaN so the plot shows gaps ----
    all_months = pd.period_range(df["Date"].min().to_period("M"),
                                 df["Date"].max().to_period("M"),
                                 freq="M")
    full = (
        out.set_index("Month")
           .reindex(all_months)  # inserts rows for months with no buys
    )
    full["Month"] = full.index
    full["MonthStart"] = full["Month"].dt.to_timestamp()
    full = full.reset_index(drop=True)

    return out, full  # out = months with buys; full = all months (NaN where no buys)


def monthly_actual_average_price(csv_path: Path) -> pd.DataFrame:
    """
    Compute the simple monthly average 'Close' across ALL trading days (not just buy days).
    Returns a DataFrame with columns: Month (period[M]), MonthStart (timestamp), actual_avg_close.
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column.")

    df["Month"] = df["Date"].dt.to_period("M")
    actual = (
        df.groupby("Month", as_index=False)
          .agg(actual_avg_close=("Close", "mean"), trading_days=("Close", "size"))
    )
    actual["MonthStart"] = actual["Month"].dt.to_timestamp()
    actual = actual.sort_values("MonthStart").reset_index(drop=True)
    return actual


def merge_buy_vs_actual(full_buy: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the 'full' buy-weighted table (includes empty months as NaN) with the actual monthly average.
    Produces a single table with both series aligned by calendar month.
    """
    merged = pd.merge(
        full_buy,
        actual[["Month", "MonthStart", "actual_avg_close"]],
        on=["Month", "MonthStart"],
        how="left",
        validate="one_to_one",
    )
    return merged


def plot_monthly_series(merged: pd.DataFrame,
                        title: str = "Monthly Averages: Buy-Weighted vs. Actual (all trading days)"):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Buy-weighted (gaps for missing buy months)
    ax.plot(merged["MonthStart"], merged["weighted_avg_close"],
            marker="o", linewidth=1.5, label="Weighted avg on buy days")

    # Actual monthly average
    ax.plot(merged["MonthStart"], merged["actual_avg_close"],
            marker="s", linewidth=1.5, label="Actual monthly avg (all days)")

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Close")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend()

    # --- Fix crowded x-axis ---
    # Show a major tick every 3 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Add minor ticks for each month (optional, makes gaps clearer)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

    # Rotate labels for readability
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    return fig



def main():
    ap = argparse.ArgumentParser(
        description="Compare monthly unit-weighted buy average price vs actual monthly average Close."
    )
    ap.add_argument("-i", "--input", default="fx_strategy_8.csv",
                    help="Input CSV with at least Date, Close, day_units.")
    ap.add_argument("-o", "--output", default="monthly_weighted_buy_price.csv",
                    help="Save table of months with buys only.")
    ap.add_argument("--output-all", default="monthly_weighted_buy_price_all_months.csv",
                    help="Save table including empty months as NaN (for plotting gaps).")
    ap.add_argument("--output-merged", default="monthly_buy_vs_actual.csv",
                    help="Save merged table with both buy-weighted and actual monthly averages.")
    ap.add_argument("--png", default="monthly_buy_vs_actual.png",
                    help="Where to save the comparison plot PNG.")
    args = ap.parse_args()

    # Buy-weighted (per-month) + a version that includes empty months
    out_buy, full_buy = monthly_weighted_buy_price(Path(args.input))
    out_buy.to_csv(args.output, index=False)       # only months with buys
    full_buy.to_csv(args.output_all, index=False)  # includes NaN rows for empty months

    # Actual monthly average over all trading days
    actual = monthly_actual_average_price(Path(args.input))

    # Merge and plot
    merged = merge_buy_vs_actual(full_buy, actual)
    merged.to_csv(args.output_merged, index=False)

    fig = plot_monthly_series(merged)
    fig.savefig(args.png, dpi=150)
    # plt.show()


if __name__ == "__main__":
    main()
