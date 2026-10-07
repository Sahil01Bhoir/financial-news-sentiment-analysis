from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "news_stock.csv"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

LABEL_MAP = {-1: "Negative", 0: "Neutral", 1: "Positive"}
TEST_SIZE = 0.2          # chronological split (last 20% of days = test)
RANDOM_STATE = 42
TOP_K_EVENTS = 3         # events per sentiment per week
LLM_MODEL = "claude-sonnet-5-5"   # any chat LLM works; see summarize_weekly.py
