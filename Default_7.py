# Jing
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# ---------- Constants ----------
EPS: float = 1e-12
TRADING_DAYS: float = 252.0
LOGGER = logging.getLogger("fx_ensemble")


# ---------- Helpers ----------
def load_prices(path: Path) -> pd.DataFrame:
    """
    Read CSV with required 'Date' and 'Close' columns (others optional), coerce types,
    drop bad rows, sort by date.
    """
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Input must contain 'Date' and 'Close' columns.")

    # Parse dates (AUD/NZD exports are often D/M/Y)
    d1 = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if d1.notna().sum() < 0.8 * len(df):  # fall back if mostly not dayfirst
        d1 = pd.to_datetime(df["Date"], errors="coerce")

    df = (
        df.assign(Date=d1, Close=pd.to_numeric(df["Close"], errors="coerce"))
        .dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Optional OHLC for ATR-based vol
    for c in ("High", "Low"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder-style RSI via EWMA approximation."""
    window = max(int(window), 1)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    alpha = 1.0 / window
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_dn = dn.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_dn + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """Rolling (x - mean)/std with ddof=0 and small EPS for stability."""
    window = max(int(window), 1)
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0)
    return (x - mu) / (sd + EPS)


def compute_vol_z(df: pd.DataFrame, vol_window: int, mode: str = "ret") -> pd.Series:
    """
    Volatility regime proxy → z-scored over a longer window.
    mode='ret': realized vol from log returns (annualized).
    mode='atr': ATR-based range vol (needs High/Low), falls back to 'ret' if missing.
    """
    vol_window = max(int(vol_window), 1)
    close = df["Close"]

    if mode == "atr" and {"High", "Low"}.issubset(df.columns):
        high, low = df["High"], df["Low"]
        prev_close = close.shift(1)
        tr = pd.DataFrame(
            {
                "hl": (high - low).abs(),
                "hc": (high - prev_close).abs(),
                "lc": (low - prev_close).abs(),
            }
        ).max(axis=1)
        vol_series = tr.rolling(vol_window).mean()  # price units
    else:
        r = np.log(close / close.shift(1))
        vol_series = r.rolling(vol_window).std(ddof=0) * np.sqrt(TRADING_DAYS)

    vol_z_win = max(vol_window * 5, vol_window + 1)  # smoother regime view
    return rolling_zscore(vol_series, vol_z_win)


def add_ensemble_features(
    df: pd.DataFrame,
    z_window: int,
    rsi_window: int,
    vol_window: int,
    vol_mode: str,
    trend_filter: bool,
    trend_window: int,
) -> pd.DataFrame:
    """Compute z, RSI, vol_z (+ optional SMA_long for regime filter)."""
    out = df.copy()
    out["z"] = rolling_zscore(out["Close"], z_window)
    out["RSI"] = compute_rsi(out["Close"], rsi_window)
    out["vol_z"] = compute_vol_z(out, vol_window, mode=vol_mode)
    if trend_filter:
        out["SMA_long"] = out["Close"].rolling(trend_window).mean()
    return out


# ---------- Signals ----------
def generate_extra_buy_signals_ensemble(
    df: pd.DataFrame,
    z_thresh: float,
    rsi_buy: float,
    w_z: float,
    w_rsi: float,
    units_per_score: float,
    max_extra_per_day: int,
    vol_z_gate: float,
    gate_high_offset: float,
    use_trend_filter: bool,
) -> pd.DataFrame:
    """
    Map oversold/mean-revert signals (z, RSI) into integer 'extra_units',
    modulated by a three-zone volatility gate and optional trend filter.
    """
    cols = ["Date", "Close", "z", "RSI", "vol_z"]
    if use_trend_filter:
        cols.append("SMA_long")
    out = df[cols].copy()

    # Normalize components ~[0..~2]
    z_depth = (-out["z"] - z_thresh) / max(1e-6, z_thresh)
    z_comp = np.clip(z_depth, 0.0, 2.0)
    rsi_comp = np.clip((rsi_buy - out["RSI"]) / max(rsi_buy, EPS), 0.0, 1.5)

    # Light hysteresis: only full score once under threshold
    entered = (out["z"] <= -z_thresh)
    hyst_mult = np.where(entered, 1.0, 0.4)
    base_score = (w_z * z_comp + w_rsi * rsi_comp) * hyst_mult

    # Volatility gate
    vol_z = out["vol_z"].to_numpy(dtype=float)
    gate_low = float(vol_z_gate)
    gate_high = gate_low + float(gate_high_offset)
    vol_mult = np.where(vol_z < gate_low, 0.6, np.where(vol_z > gate_high, 0.5, 1.0))

    # Optional trend/regime filter: only buy when price >= SMA_long
    if use_trend_filter and "SMA_long" in out.columns:
        trend_mult = (out["Close"] >= out["SMA_long"]).astype(float).to_numpy()
    else:
        trend_mult = 1.0

    final_score = base_score * vol_mult * trend_mult

    # Score → integer units with per-day cap
    units = np.floor(final_score * float(units_per_score)).astype(int)
    units = np.clip(units, 0, int(max_extra_per_day))

    out = out.assign(
        extra_units=units.astype(int),
        rank_score=out["z"],     # lower = better
        final_score=final_score, # diagnostics
    )
    return out[["Date", "Close", "extra_units", "rank_score", "final_score"]]


# ---------- Global month index & bounds ----------
def month_index_series(dates: pd.Series) -> pd.Series:
    """Monotone 1,2,3,… month index across the whole sample."""
    periods = dates.dt.to_period("M")
    ordered = pd.PeriodIndex(periods.dropna().unique()).sort_values()
    mapping = {p: i + 1 for i, p in enumerate(ordered)}
    return periods.map(mapping)


def bounds_for_month_index(m_idx: int, margin: int = 2) -> Tuple[int, int, int]:
    """Ideal = 2*m, bounds = ideal ± margin."""
    ideal = 2 * int(m_idx)
    lower = max(0, ideal - margin)
    upper = ideal + margin
    return lower, ideal, upper


# ---------- Enforce monthly **position** range (NO base drip) ----------
def enforce_global_monthly_position_range_no_base(
    df: pd.DataFrame,
    margin: int = 2,
    max_position_units: Optional[float] = None,
    price_for_tilt_window: int = 60,
) -> pd.DataFrame:
    """
    Enforce that cumulative *position_units* at each month-end lands within
    [ideal - margin, ideal + margin], with a mild price-aware tilt.
    """
    need_cols = {"Date", "Close", "extra_units", "rank_score"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"enforce_range: missing columns {missing}")

    out = df.copy().sort_values("Date").reset_index(drop=True)
    out["month_index"] = month_index_series(out["Date"])

    roll_mean = out["Close"].rolling(
        price_for_tilt_window, min_periods=max(1, price_for_tilt_window // 2)
    ).mean()

    pos_cum = 0.0

    for m_idx in sorted(out["month_index"].dropna().unique()):
        idx = list(out.index[out["month_index"] == m_idx])
        m = int(m_idx)

        extra = out.loc[idx, "extra_units"].astype(float).to_numpy()
        scores = out.loc[idx, "rank_score"].to_numpy()
        month_close = out.loc[idx, "Close"].to_numpy()

        extra_adds_raw = float(np.nansum(extra))

        ideal = 2.0 * m

        # Price-aware tilt vs. rolling mean
        p_month = float(np.nanmean(month_close))
        slice_mu = roll_mean.iloc[idx].to_numpy(dtype=float)
        p_mu = float(np.nanmean(slice_mu)) if np.isfinite(slice_mu).any() else np.nan
        if np.isfinite(p_month) and np.isfinite(p_mu) and p_mu > EPS:
            tilt = np.clip((p_mu - p_month) / (0.5 * p_mu + EPS), -0.5, 0.5)
        else:
            tilt = 0.0

        ideal_tilted = ideal + margin * tilt * 2.0
        lower_total = max(0.0, ideal_tilted - margin)
        upper_total = ideal_tilted + margin

        pos_start = pos_cum
        min_add = max(0.0, lower_total - pos_start)
        max_add = max(0.0, upper_total - pos_start)

        # If already above the upper bound → suppress extras this month
        if pos_start > upper_total + EPS:
            extra[:] = 0.0
        else:
            # Trim if over-allocating
            if extra_adds_raw > max_add + EPS:
                cands = [(i_local, scores[i_local]) for i_local in range(len(idx)) if extra[i_local] > 0]
                cands.sort(key=lambda x: x[1], reverse=True)  # worst (less negative z) removed first
                to_remove = extra_adds_raw - max_add
                j = 0
                while to_remove > EPS and cands:
                    i_local = cands[j % len(cands)][0]
                    if extra[i_local] > 0:
                        extra[i_local] -= 1.0
                        to_remove -= 1.0
                    j += 1

            # Top up if under-allocating
            elif extra_adds_raw < min_add - EPS:
                need = min_add - extra_adds_raw
                order = sorted(range(len(idx)), key=lambda i_local: scores[i_local])  # best (most negative z) first
                k = 0
                while need > EPS and k < len(order):
                    i_local = order[k]
                    extra[i_local] += 1.0
                    need -= 1.0
                    k += 1
                # If still short and order isn't empty, round-robin
                k = 0
                while need > EPS and order:
                    i_local = order[k % len(order)]
                    extra[i_local] += 1.0
                    need -= 1.0
                    k += 1

            # Re-check cap after adjustments
            total_adds = float(np.nansum(extra))
            if pos_start + total_adds > upper_total + EPS:
                over = (pos_start + total_adds) - upper_total
                cands = [(i_local, scores[i_local]) for i_local in range(len(idx)) if extra[i_local] > 0]
                cands.sort(key=lambda x: x[1], reverse=True)
                j = 0
                while over > EPS and cands:
                    i_local = cands[j % len(cands)][0]
                    if extra[i_local] > 0:
                        extra[i_local] -= 1.0
                        over -= 1.0
                    j += 1

        out.loc[idx, "extra_units"] = extra
        pos_cum += float(np.nansum(extra))

        # Global hard cap
        if (max_position_units is not None) and (pos_cum > max_position_units + EPS):
            rollback = pos_cum - max_position_units
            cands = [(i_local, -scores[i_local]) for i_local in range(len(idx)) if extra[i_local] > 0]
            cands.sort(key=lambda x: x[1])  # remove worst first (less negative z)
            j = 0
            while rollback > EPS and cands:
                i_local = cands[j % len(cands)][0]
                if extra[i_local] > 0:
                    extra[i_local] -= 1.0
                    rollback -= 1.0
                j += 1
            out.loc[idx, "extra_units"] = extra
            pos_cum = float(max_position_units)

    out["day_units"] = out["extra_units"].astype(float)
    out["position_units"] = out["day_units"].cumsum()
    return out.drop(columns=["month_index"])


# ---------- Plotting ----------
def _finite(a: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr[np.isfinite(arr)]


def _set_padded_ylim(ax: plt.Axes, values: np.ndarray, pad_frac: float) -> None:
    vals = _finite(values)
    if vals.size == 0:
        return
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        span = abs(vmax) if vmax != 0 else 1.0
        vmin, vmax = vmin - 0.5 * span * pad_frac, vmax + 0.5 * span * pad_frac
    else:
        pad = (vmax - vmin) * pad_frac
        vmin, vmax = vmin - pad, vmax + pad
    ax.set_ylim(vmin, vmax)


def plot_strategy_panels(
    df: pd.DataFrame,
    z_thresh: float,
    vol_low: float,
    vol_high: float,
    png_path: Path,
) -> None:
    """Three panels: Price+buys, vol_z with gates, z-score with threshold."""
    dplot = df.sort_values("Date").copy()
    extra_col = pd.to_numeric(dplot.get("extra_units", 0), errors="coerce").fillna(0).astype(int)
    dplot["buy_signal"] = (extra_col > 0).astype(int)
    mask = dplot["buy_signal"] == 1

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    # Panel 1: Price + buy markers
    ax0 = axes[0]
    ax0.plot(dplot["Date"], dplot["Close"], linewidth=1.5, color="black", label="AUD/NZD Close")
    if mask.any():
        sizes = 36 + 12 * np.log1p(pd.to_numeric(dplot.loc[mask, "extra_units"], errors="coerce").fillna(0).astype(float))
        ax0.scatter(
            dplot.loc[mask, "Date"],
            dplot.loc[mask, "Close"],
            s=sizes, marker="o", facecolors="orange", edgecolors="k",
            linewidths=0.6, alpha=0.95, zorder=5, label="Buy (extra units)"
        )
    ax0.set_ylabel("Price")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left")
    ax0.set_title("AUD/NZD — Ensemble Buy-Low Signals (No Base Drip)")
    _set_padded_ylim(ax0, dplot["Close"].to_numpy(dtype=float), pad_frac=0.06)

    # Panel 2: vol_z + gates
    ax1 = axes[1]
    if "vol_z" in dplot.columns:
        vz = pd.to_numeric(dplot["vol_z"], errors="coerce")
        if np.isfinite(vz).any():
            ax1.plot(dplot["Date"], vz, linewidth=1.0, label="vol_z")
            ax1.axhline(vol_low, linestyle="--", linewidth=1.0, label=f"vol low ({vol_low:g})")
            ax1.axhline(vol_high, linestyle="--", linewidth=1.0, label=f"vol high ({vol_high:g})")
            ax1.fill_between(dplot["Date"], vz, vol_low, where=(vz < vol_low), interpolate=True, alpha=0.12)
            ax1.fill_between(dplot["Date"], vz, vol_high, where=(vz > vol_high), interpolate=True, alpha=0.12)
            ax1.set_ylabel("vol z")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="upper left")
            _set_padded_ylim(ax1, np.concatenate([vz[np.isfinite(vz)].to_numpy(), np.array([vol_low, vol_high])]), 0.10)
        else:
            ax1.text(0.01, 0.5, "No vol_z data", transform=ax1.transAxes, va="center")
            ax1.set_axis_off()
    else:
        ax1.text(0.01, 0.5, "No vol_z column", transform=ax1.transAxes, va="center")
        ax1.set_axis_off()

    # Panel 3: z + threshold & buys
    ax2 = axes[2]
    if "z" in dplot.columns:
        z = pd.to_numeric(dplot["z"], errors="coerce")
        has_z = np.isfinite(z).any()
        if has_z:
            ax2.plot(dplot["Date"], z, linewidth=1.0, label="z-score(price)")
        ax2.axhline(-z_thresh, linewidth=1.0, linestyle="--", label=f"z thresh (-{z_thresh:g})")
        ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
        if has_z:
            ax2.fill_between(dplot["Date"], z, -z_thresh, where=(z <= -z_thresh), interpolate=True, alpha=0.15)
            if mask.any():
                ystem = np.minimum(z[mask].to_numpy(dtype=float), -z_thresh)
                ax2.vlines(dplot.loc[mask, "Date"], ystem, -z_thresh, linewidth=1.0, alpha=0.85)
                ax2.scatter(dplot.loc[mask, "Date"], ystem, s=18, facecolors="orange", edgecolors="k", zorder=3)
        ax2.set_ylabel("z-score")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper left")
    else:
        ax2.text(0.01, 0.5, "No z column to plot", transform=ax2.transAxes, va="center")
        ax2.set_axis_off()

    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------- CLI ----------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Buy-low ensemble with NO daily base drip + global monthly POSITION envelope (price-aware)."
    )
    ap.add_argument("--input", "-i", default="NZD_AUD_10y.csv", help="CSV with Date, Close (and optional High, Low for ATR).")
    ap.add_argument("--output", "-o", default="fx_strategy_8.csv", help="Output CSV path.")
    ap.add_argument("--plot-file", default="fx_price_8.png", help="Output PNG path.")

    # Envelope
    ap.add_argument("--range-margin", type=int, default=4, help="Margin around ideal position at month-end (ideal±margin).")
    ap.add_argument("--nzd-per-unit", type=float, default=900000.0, help="NZD exchanged per buy unit.")
    ap.add_argument("--max-position-units", type=float, default=None, help="Optional hard cap on cumulative units.")

    # Features / thresholds
    ap.add_argument("--z-window", type=int, default=60, help="Window for price z-score.")
    ap.add_argument("--z-thresh", type=float, default=1.0, help="Buy pressure when z <= -z_thresh.")
    ap.add_argument("--rsi-window", type=int, default=10, help="RSI window.")
    ap.add_argument("--rsi-buy", type=float, default=38.0, help="RSI level considered oversold (<= level).")
    ap.add_argument("--vol-window", type=int, default=30, help="Window for vol measure.")
    ap.add_argument("--vol-mode", choices=["ret", "atr"], default="ret", help="Volatility proxy: return vol or ATR-based.")
    ap.add_argument("--vol-z-gate", type=float, default=1.2, help="Volatility z-score low gate.")
    ap.add_argument("--gate-high-offset", type=float, default=0.8, help="Offset above vol_z_gate for 'panic' zone.")

    # Optional regime filter
    ap.add_argument("--trend-filter", action="store_true", help="Only buy when Close >= SMA_long.")
    ap.add_argument("--trend-window", type=int, default=100, help="Window for SMA_long if trend-filter is enabled.")

    # Weights & mapping to units
    ap.add_argument("--w-z", type=float, default=1.0, help="Weight for z-score component.")
    ap.add_argument("--w-rsi", type=float, default=0.4, help="Weight for RSI oversold component.")
    ap.add_argument("--units-per-score", type=float, default=1.5, help="Multiplier from score to units before floor.")
    ap.add_argument("--max-extra-per-day", type=int, default=3, help="Cap on extra units per day.")

    # Execution lag
    ap.add_argument("--lag-days", type=int, default=0, help="Execution lag for extras (days).")

    # PnL/EQ params
    ap.add_argument("--tcost-bps", type=float, default=0.5, help="Per-buy cost in bps on AUD notional.")
    ap.add_argument("--seed-aud", type=float, default=100000.0, help="Initial equity seed for stable returns.")

    # Logging
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable info logging.")
    return ap.parse_args(argv)


# ---------- Main ----------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    LOGGER.info("Starting run with args: %s", vars(args))

    ipath = Path(args.input)
    if not ipath.exists():
        LOGGER.error("Input not found: %s", ipath)
        sys.exit(1)

    # Load & features
    px = load_prices(ipath)
    px = add_ensemble_features(
        px,
        z_window=args.z_window,
        rsi_window=args.rsi_window,
        vol_window=args.vol_window,
        vol_mode=args.vol_mode,
        trend_filter=args.trend_filter,
        trend_window=args.trend_window,
    )

    # Drop warm-up NaNs
    req_cols = ["z", "RSI", "vol_z"] + (["SMA_long"] if args.trend_filter else [])
    px = px.dropna(subset=req_cols).reset_index(drop=True)

    # Ensemble extras only (no base drip)
    extra = generate_extra_buy_signals_ensemble(
        px,
        z_thresh=args.z_thresh,
        rsi_buy=args.rsi_buy,
        w_z=args.w_z,
        w_rsi=args.w_rsi,
        units_per_score=args.units_per_score,
        max_extra_per_day=args.max_extra_per_day,
        vol_z_gate=args.vol_z_gate,
        gate_high_offset=args.gate_high_offset,
        use_trend_filter=args.trend_filter,
    )
    LOGGER.info("Raw extra signal days (pre-lag): %d", int((extra["extra_units"] > 0).sum()))

    # Execution lag
    if args.lag_days > 0:
        extra["extra_units"] = extra["extra_units"].shift(args.lag_days).fillna(0).astype(int)
    LOGGER.info("Extra signal days (post-lag): %d", int((extra["extra_units"] > 0).sum()))

    # Enforce monthly **position** range
    sig_adj = enforce_global_monthly_position_range_no_base(
        extra[["Date", "Close", "extra_units", "rank_score"]].copy(),
        margin=args.range_margin,
        max_position_units=args.max_position_units,
        price_for_tilt_window=60,

    )
    LOGGER.info("Kept extra signal days (after range): %d", int((sig_adj["extra_units"] > 0).sum()))

    # Merge & compute flows
    out = px.merge(
        sig_adj[["Date", "extra_units", "day_units", "position_units"]],
        on="Date",
        how="left",
    ).fillna({"extra_units": 0, "day_units": 0.0})

    out["buy_signal"] = (pd.to_numeric(out["extra_units"], errors="coerce").fillna(0).astype(int) > 0).astype(int)

    # Notional flows
    out["NZD_spent"] = out["day_units"] * args.nzd_per_unit
    out["AUD_bought_gross"] = out["NZD_spent"] * out["Close"]

    # Transaction cost
    # tcost = float(args.tcost_bps) * 1e-4
    # out["AUD_bought"] = out["AUD_bought_gross"] * (1.0 - tcost)
    # No transaction cost applied
    out["AUD_bought"] = out["AUD_bought_gross"]

    # Cumulative flows & equity
    out["strategy_returns"] = out["AUD_bought"]
    out["cumulative_returns"] = out["AUD_bought"].cumsum()
    out["cash_aud"] = (-out["AUD_bought"]).cumsum()
    out["portfolio_value_aud"] = out["position_units"] * args.nzd_per_unit * out["Close"]
    out["equity_aud"] = float(args.seed_aud) + out["cash_aud"] + out["portfolio_value_aud"]
    out["equity_return"] = (
        out["equity_aud"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Save CSV
    opath = Path(args.output)
    out.to_csv(opath, index=False)

    # Plot
    ppath = Path(args.plot_file)
    df_plot = out.copy()
    for col in ("z", "buy_signal", "vol_z"):
        if col not in df_plot.columns:
            df_plot[col] = np.nan
    vol_low = args.vol_z_gate
    vol_high = args.vol_z_gate + args.gate_high_offset
    plot_strategy_panels(df_plot, z_thresh=args.z_thresh, vol_low=vol_low, vol_high=vol_high, png_path=ppath)

    # Month summary/check
    out["month_index"] = month_index_series(out["Date"])
    month_sum = (
        out.groupby("month_index", dropna=True)[["extra_units", "day_units"]]
        .sum()
        .reset_index()
        .rename(columns={"day_units": "total_units"})
    )

    out["month_end"] = out["Date"].dt.to_period("M")
    month_end_pos = out.groupby("month_end")["position_units"].last().reset_index(name="pos_end")
    month_end_pos["month_index"] = month_index_series(month_end_pos["month_end"].dt.to_timestamp())

    check = month_end_pos.merge(month_sum, on="month_index", how="left")
    bounds = check["month_index"].apply(lambda m: pd.Series(bounds_for_month_index(int(m), args.range_margin)))
    check[["lower", "ideal", "upper"]] = bounds

    print(f"CSV saved:  {opath}")
    print(f"Plot saved: {ppath}")
    print("Global month-end position check (no base drip):")
    print(
        check[
            ["month_index", "pos_end", "lower", "ideal", "upper", "extra_units", "total_units"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
