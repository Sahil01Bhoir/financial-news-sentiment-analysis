"""Step 4: link sentiment to price moves and test a simple signal."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIG_DIR, REPORT_DIR

def lagged_correlation(df, max_lag=5):
    """corr(sentiment_t, return_{t+lag}); lag 0 = same day."""
    rows = [dict(lag=l, corr=df["sentiment_mean"].corr(df["return"].shift(-l))) for l in range(0, max_lag + 1)]
    out = pd.DataFrame(rows)
    plt.figure(figsize=(6, 3.5))
    plt.bar(out["lag"], out["corr"], color="#1f77b4"); plt.axhline(0, color="k", lw=.5)
    plt.xlabel("Lag (days after news)"); plt.ylabel("Correlation"); plt.title("Sentiment -> future returns")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_lagged_correlation.png", dpi=150); plt.close()
    return out

def sentiment_shift_events(df, thr=0.25):
    """Days where 7-day sentiment moved sharply; what did price do over the next 5 days?"""
    d = df.copy()
    d["shift"] = d["sentiment_ma7"].diff(3)
    d["fwd_5d"] = d["Close"].shift(-5) / d["Close"] - 1
    up, down = d[d["shift"] >= thr], d[d["shift"] <= -thr]
    return pd.DataFrame({"n_events": [len(up), len(down)],
                         "avg_fwd_5d_return": [up["fwd_5d"].mean(), down["fwd_5d"].mean()]},
                        index=["sentiment surge", "sentiment drop"])

def backtest(df, signal_col="sentiment_mean", cost=0.0005, band=0.1):
    """Long if yesterday's mean sentiment > +band, short if < -band, else flat.
    Swap signal_col for model predictions to evaluate out-of-sample."""
    d = df.dropna(subset=["return"]).copy()
    sig = d[signal_col]
    pos = pd.Series(np.where(sig.abs() < band, 0, np.sign(sig)), index=d.index).shift(1).fillna(0)
    d["strategy"] = pos * d["return"] - cost * pos.diff().abs().fillna(0)
    d["cum_strategy"] = (1 + d["strategy"]).cumprod()
    d["cum_buyhold"] = (1 + d["return"]).cumprod()
    sharpe = lambda r: np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan
    stats = pd.Series({"strategy_total_return": d["cum_strategy"].iloc[-1] - 1,
                       "buyhold_total_return": d["cum_buyhold"].iloc[-1] - 1,
                       "strategy_sharpe": sharpe(d["strategy"]),
                       "buyhold_sharpe": sharpe(d["return"])})
    plt.figure(figsize=(9, 4))
    plt.plot(d["Date"], d["cum_buyhold"], label="Buy & hold")
    plt.plot(d["Date"], d["cum_strategy"], label="Sentiment strategy")
    plt.legend(); plt.title("Backtest (illustrative, includes transaction cost)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "07_backtest.png", dpi=150); plt.close()
    return stats

def run_insights(df):
    lag = lagged_correlation(df)
    shifts = sentiment_shift_events(df)
    stats = backtest(df)
    print("\nLagged correlation:\n", lag.round(3))
    print("\nSentiment shifts -> next 5-day return:\n", shifts.round(4))
    print("\nBacktest:\n", stats.round(3))
    with open(REPORT_DIR / "insights.md", "w") as f:
        f.write("# Insights\n\n## Lagged correlation\n" + lag.round(3).to_string(index=False) +
                "\n\n## Sentiment shifts\n" + shifts.round(4).to_string() +
                "\n\n## Backtest\n" + stats.round(3).to_string() + "\n")
