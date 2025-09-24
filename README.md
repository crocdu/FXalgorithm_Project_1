# Jing — FX Buy-Low Ensemble Strategy  

## Overview  
This script implements a **buy-low ensemble strategy** for AUD/NZD foreign exchange.  
It generates additional buy signals based on mean-reversion indicators (z-score and RSI),  
applies volatility gating and optional trend filters, and enforces global monthly **position envelopes**  
to control risk and allocation.  

The strategy outputs both **CSV trade signals** and **visual plots** to help analyze performance.  

## Approach  

### Data Sources  
- **Input CSV**: Historical AUD/NZD prices with at least:  
  - `Date`  
  - `Close`  
  - Optional: `High`, `Low` (for ATR-based volatility).  

### Feature Engineering  
- **Z-score** of price (rolling mean and std).  
- **RSI (Relative Strength Index)** using Wilder’s smoothing.  
- **Volatility z-score** based on returns or ATR.  
- Optional **long-term SMA** filter to only buy in uptrends.  

### Signal Generation  
- Oversold detection via `z` and `RSI`.  
- Weighted combination of z-score and RSI components.  
- Volatility gating with three regimes: low, normal, high.  
- Optional **trend filter** (price above SMA).  
- Signals mapped to discrete **extra units** with daily caps.  

### Position Envelope  
- Monthly position targets enforced as `2 × month_index ± margin`.  
- Adjusts buy signals to remain within monthly bounds.  
- Optional global cap on maximum position size.  

### Outputs  
- **CSV file** with enriched features, signals, positions, and equity.  
- **PNG plot** with three panels:  
  1. Price with buy signals.  
  2. Volatility z-score with gates.  
  3. Z-score with thresholds and buy markers.  
- Console summary of monthly envelope checks.  

## Key Features of the Script  
- Loads, cleans, and parses FX price data.  
- Computes RSI, z-score, and volatility regime proxies.  
- Generates ensemble-based buy signals.  
- Enforces monthly allocation rules with tilt adjustments.  
- Saves enriched signals to CSV.  
- Produces multi-panel diagnostic plots.  
- Flexible CLI arguments for customization.  

## File Structure  
- **jing.py** (or `Jing`)  
  Main script containing data loading, feature engineering, signal generation,  
  position enforcement, and plotting logic.  

- **Input Data (user-provided)**  
  - `NZD_AUD_10y.csv`: Historical AUD/NZD prices with `Date` and `Close` columns.  

- **Outputs**  
  - `fx_strategy_8.csv`: CSV with signals, allocations, and equity.  
  - `fx_price_8.png`: Diagnostic plot with signals and indicators.  

## How to Run  
1. Run from the command line with defaults:  
   ```bash
   python jing.py -i NZD_AUD_10y.csv -o fx_strategy_8.csv --plot-file fx_price_8.png
