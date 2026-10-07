"""Step 1: cleaning, feature engineering and EDA."""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, FIG_DIR, LABEL_MAP

URL = re.compile(r"https?://\S+")
NON_ALNUM = re.compile(r"[^a-z0-9\s$%.]")

def clean_text(t: str) -> str:
    t = URL.sub(" ", str(t).lower())
    t = NON_ALNUM.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day: prices are constant within a day, sentiment is aggregated over that day's articles."""
    g = df.groupby("Date")
    daily = g[PRICE_COLS].first()
    daily["sentiment_mean"] = g["Label"].mean()
    daily["n_articles"] = g.size()
    daily = daily.reset_index().sort_values("Date").reset_index(drop=True)
    daily["return"] = daily["Close"].pct_change()
    daily["next_return"] = daily["return"].shift(-1)   # what sentiment is supposed to predict
    daily["range_pct"] = (daily["High"] - daily["Low"]) / daily["Open"]
    daily["volatility_5d"] = daily["return"].rolling(5).std()
    daily["vol_change"] = daily["Volume"].pct_change()
    daily["sentiment_ma7"] = daily["sentiment_mean"].rolling(7, min_periods=3).mean()
    return daily

def load_and_clean(path=DATA_PATH) -> pd.DataFrame:
    """Article-level frame (one row per news item) enriched with the day's price features."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "News"]).dropna(subset=["Date", "News", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["clean_news"] = df["News"].map(clean_text)
    df = df[df["clean_news"].str.len() > 0].copy()
    df["Label"] = df["Label"].astype(int)
    daily = build_daily(df)
    keep = ["Date", "return", "next_return", "range_pct", "volatility_5d", "vol_change", "sentiment_mean", "sentiment_ma7"]
    return df.merge(daily[keep], on="Date", how="left").reset_index(drop=True)

def run_eda(df: pd.DataFrame, daily: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    names = df["Label"].map(LABEL_MAP)
    order = ["Negative", "Neutral", "Positive"]

    # 1. class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x=names, order=order, hue=names, hue_order=order, legend=False,
                  palette={"Negative": "#d62728", "Neutral": "#7f7f7f", "Positive": "#2ca02c"})
    plt.title("Sentiment distribution"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(FIG_DIR / "01_label_distribution.png", dpi=150); plt.close()

    # 2. price vs rolling sentiment
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(daily["Date"], daily["Close"], color="#1f77b4"); ax[0].set_ylabel("Close")
    ax[1].plot(daily["Date"], daily["sentiment_ma7"], color="#ff7f0e"); ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_ylabel("7-day avg sentiment")
    fig.suptitle("Price vs rolling news sentiment"); fig.tight_layout()
    fig.savefig(FIG_DIR / "02_price_vs_sentiment.png", dpi=150); plt.close(fig)

    # 3. return and volume by sentiment
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x=names, y=df["next_return"], order=order, ax=ax[0])
    ax[0].set_title("Next-day return by sentiment"); ax[0].set_xlabel("")
    sns.boxplot(x=names, y=df["Volume"], order=order, ax=ax[1])
    ax[1].set_title("Volume by sentiment"); ax[1].set_xlabel("")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_return_volume_by_sentiment.png", dpi=150); plt.close(fig)

    # 4. correlation heatmap
    cols = ["sentiment_mean", "return", "next_return", "range_pct", "volatility_5d", "vol_change", "n_articles"]
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(daily[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix"); plt.tight_layout()
    plt.savefig(FIG_DIR / "04_correlations.png", dpi=150); plt.close()

    summary = df.groupby(names).agg(articles=("Label", "size"), avg_next_return=("next_return", "mean"),
                                    avg_volume=("Volume", "mean"), avg_range=("range_pct", "mean"))
    print("\nEDA summary by sentiment:\n", summary.round(4))
    return summary
