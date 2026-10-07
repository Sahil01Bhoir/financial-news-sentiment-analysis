"""Step 3: weekly summaries of the most impactful positive and negative events.
Impact score = |next-day return| x (1 + volume z-score), restricted to the matching sentiment.
Uses an LLM when ANTHROPIC_API_KEY is set; otherwise falls back to a template summary."""
import os
import pandas as pd
from config import REPORT_DIR, TOP_K_EVENTS, LLM_MODEL

def pick_events(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    vol = df.drop_duplicates("Date")["Volume"]          # one volume per trading day
    d["vol_z"] = (d["Volume"] - vol.mean()) / vol.std()
    # tiny length-based tie-breaker: articles from the same day share the same price move
    d["impact"] = (d["next_return"].abs().fillna(0) * (1 + d["vol_z"].clip(lower=0))
                   + 1e-9 * d["News"].str.len())
    d["week"] = d["Date"].dt.to_period("W-SUN").dt.start_time
    d = d[d["Label"] != 0].copy()
    d["rank"] = d.groupby(["week", "Label"])["impact"].rank(ascending=False, method="first")
    return d[d["rank"] <= TOP_K_EVENTS].sort_values(["week", "Label", "rank"], ascending=[True, False, True])

def _llm_summary(week, pos, neg, avg_sent, week_ret):
    import anthropic
    client = anthropic.Anthropic()
    prompt = (f"You are an equity research analyst. Week starting {week:%Y-%m-%d}: "
              f"avg sentiment {avg_sent:+.2f}, weekly return {week_ret:+.2%}.\n\n"
              "Positive headlines:\n- " + "\n- ".join(pos or ["none"]) +
              "\n\nNegative headlines:\n- " + "\n- ".join(neg or ["none"]) +
              "\n\nWrite a 3-4 sentence summary of the key drivers, then one line: "
              "'Outlook: bullish/neutral/bearish' with a brief reason. Do not invent facts.")
    msg = client.messages.create(model=LLM_MODEL, max_tokens=300,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text.strip()

def _template_summary(pos, neg, avg_sent, week_ret):
    tone = "bullish" if avg_sent > 0.15 else "bearish" if avg_sent < -0.15 else "mixed"
    return (f"Tone was {tone} (avg sentiment {avg_sent:+.2f}, return {week_ret:+.2%}). "
            f"Top positive: {pos[0] if pos else 'none'}. Top negative: {neg[0] if neg else 'none'}.")

def build_weekly_report(df: pd.DataFrame, max_weeks=12) -> pd.DataFrame:
    events = pick_events(df)
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    d = df.assign(week=df["Date"].dt.to_period("W-SUN").dt.start_time)
    weekly = d.groupby("week").agg(avg_sent=("Label", "mean"), start=("Close", "first"), end=("Close", "last"))
    rows = []
    for week, g in list(events.groupby("week"))[-max_weeks:]:
        pos = g.loc[g["Label"] == 1, "News"].tolist()
        neg = g.loc[g["Label"] == -1, "News"].tolist()
        w = weekly.loc[week]; ret = w["end"] / w["start"] - 1
        try:
            summary = (_llm_summary(week, pos, neg, w["avg_sent"], ret) if use_llm
                       else _template_summary(pos, neg, w["avg_sent"], ret))
        except Exception as e:
            print(f"[LLM fallback] {e}")
            summary = _template_summary(pos, neg, w["avg_sent"], ret)
        rows.append(dict(week=week.date(), avg_sentiment=round(w["avg_sent"], 3),
                         weekly_return=round(ret, 4), summary=summary))
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "weekly_summaries.csv", index=False)
    with open(REPORT_DIR / "weekly_summaries.md", "w") as f:
        f.write("# Weekly Impactful Events\n\n")
        for r in rows:
            f.write(f"## Week of {r['week']}  (sentiment {r['avg_sentiment']:+.2f}, return {r['weekly_return']:+.2%})\n{r['summary']}\n\n")
    print(f"Saved {len(out)} weekly summaries ({'LLM' if use_llm else 'template'} mode)")
    return out
