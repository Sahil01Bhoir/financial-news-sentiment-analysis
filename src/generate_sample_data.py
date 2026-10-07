"""Creates a synthetic dataset with the same schema as the real one.
Replace data/news_stock.csv with real data (Date, News, Open, High, Low, Close, Volume, Label)."""
import numpy as np
import pandas as pd
from config import DATA_PATH

POS = ["{c} beats quarterly earnings expectations as revenue surges",
       "{c} announces record profit and raises full-year guidance",
       "Analysts upgrade {c} citing strong demand and margin expansion",
       "{c} wins major contract, shares rally"]
NEG = ["{c} misses earnings estimates amid weak demand",
       "{c} cuts guidance as costs and inflation hit margins",
       "Regulators open probe into {c}, shares slump",
       "{c} announces layoffs after disappointing quarter"]
NEU = ["{c} to hold annual general meeting next month",
       "{c} appoints new chief technology officer",
       "{c} reports results in line with expectations",
       "{c} schedules investor conference call"]

def generate(n_days=600, seed=42, company="Acme Corp"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    labels = rng.choice([-1, 0, 1], size=n_days, p=[0.28, 0.34, 0.38])
    templates = {1: POS, 0: NEU, -1: NEG}
    news = [rng.choice(templates[l]).format(c=company) for l in labels]
    # next-day return is weakly driven by sentiment (so insights are non-trivial)
    shifted = np.r_[0, labels[:-1]]
    ret = 0.004 * shifted + rng.normal(0, 0.012, n_days)
    close = 100 * np.cumprod(1 + ret)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_days))
    volume = (1e6 * (1 + 0.5 * np.abs(labels) + rng.normal(0, 0.2, n_days))).astype(int)
    return pd.DataFrame(dict(Date=dates, News=news, Open=open_.round(2), High=high.round(2),
                             Low=low.round(2), Close=close.round(2), Volume=volume, Label=labels))

if __name__ == "__main__":
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved synthetic data -> {DATA_PATH} ({len(df)} rows)")
