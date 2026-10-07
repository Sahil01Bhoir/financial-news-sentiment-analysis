import sys
sys.path.insert(0, "src")
from config import DATA_PATH, FIG_DIR, REPORT_DIR
from generate_sample_data import generate
from preprocess_eda import load_and_clean, build_daily, run_eda
from embeddings_models import run_models
from summarize_weekly import build_weekly_report
from insights import run_insights

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("No data found - generating synthetic sample data.")
        generate().to_csv(DATA_PATH, index=False)
    df = load_and_clean()            # article-level (one row per news item)
    daily = build_daily(df)          # day-level (one row per trading day)
    print(f"{len(df)} articles across {len(daily)} trading days "
          f"({daily['Date'].min():%Y-%m-%d} to {daily['Date'].max():%Y-%m-%d})")
    run_eda(df, daily)
    run_models(df)                   # pass e.g. ["TF-IDF", "SentenceTransformer"] to limit
    build_weekly_report(df)
    run_insights(daily)
