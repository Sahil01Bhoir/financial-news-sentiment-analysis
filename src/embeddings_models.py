"""Step 2: embeddings (Word2Vec, GloVe, Sentence Transformers) + sentiment classifiers."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from config import FIG_DIR, REPORT_DIR, TEST_SIZE, RANDOM_STATE

def chrono_split(df):
    """Split by date so no trading day appears in both train and test."""
    dates = np.sort(df["Date"].unique())
    cut = dates[int(len(dates) * (1 - TEST_SIZE))]
    return df[df["Date"] < cut], df[df["Date"] >= cut]

def _avg_vectors(tokens_list, lookup, dim):
    out = np.zeros((len(tokens_list), dim), dtype=np.float32)
    for i, toks in enumerate(tokens_list):
        vecs = [lookup[t] for t in toks if t in lookup]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out

def word2vec_embed(train_txt, test_txt, dim=100):
    from gensim.models import Word2Vec
    tr = [t.split() for t in train_txt]; te = [t.split() for t in test_txt]
    w2v = Word2Vec(tr, vector_size=dim, window=5, min_count=1, workers=2, epochs=30, seed=RANDOM_STATE)
    return _avg_vectors(tr, w2v.wv, dim), _avg_vectors(te, w2v.wv, dim)

def glove_embed(train_txt, test_txt):
    import gensim.downloader as api
    glove = api.load("glove-wiki-gigaword-100")      # ~130 MB, downloaded once
    tr = [t.split() for t in train_txt]; te = [t.split() for t in test_txt]
    return _avg_vectors(tr, glove, 100), _avg_vectors(te, glove, 100)

def sbert_embed(train_txt, test_txt, model="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    return (m.encode(list(train_txt), show_progress_bar=False),
            m.encode(list(test_txt), show_progress_bar=False))

def tfidf_embed(train_txt, test_txt):
    """Sparse baseline to compare the learned embeddings against."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    return v.fit_transform(train_txt), v.transform(test_txt)

EMBEDDERS = {"TF-IDF": tfidf_embed, "Word2Vec": word2vec_embed, "GloVe": glove_embed, "SentenceTransformer": sbert_embed}
CLASSIFIERS = {
    "LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                   random_state=RANDOM_STATE, n_jobs=-1),
}

def run_models(df, embedders=None):
    """Returns (results_df, best_info). Skips an embedder if its library/model is unavailable."""
    train, test = chrono_split(df)
    y_tr, y_te = train["Label"].values, test["Label"].values
    print(f"\nTrain: {len(train)} articles up to {train['Date'].max():%Y-%m-%d} | "
          f"Test: {len(test)} articles from {test['Date'].min():%Y-%m-%d}")
    rows, store = [], {}
    for emb_name in (embedders or EMBEDDERS):
        try:
            X_tr, X_te = EMBEDDERS[emb_name](train["clean_news"], test["clean_news"])
        except Exception as e:
            print(f"[skip] {emb_name}: {e}"); continue
        for clf_name, make in CLASSIFIERS.items():
            clf = make().fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            rows.append(dict(embedding=emb_name, model=clf_name,
                             accuracy=accuracy_score(y_te, pred),
                             macro_f1=f1_score(y_te, pred, average="macro")))
            store[(emb_name, clf_name)] = pred
    if not rows:
        print("No embedding method could run."); return pd.DataFrame(), None
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    results.to_csv(REPORT_DIR / "model_results.csv", index=False)
    print("\nModel comparison (chronological test set):\n", results.round(3))

    best = results.iloc[0]
    pred = store[(best["embedding"], best["model"])]
    print(f"\nBest: {best['embedding']} + {best['model']}")
    print(classification_report(y_te, pred, labels=[-1, 0, 1],
                                target_names=["Negative", "Neutral", "Positive"], zero_division=0))
    cm = confusion_matrix(y_te, pred, labels=[-1, 0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.xticks(range(3), ["Neg", "Neu", "Pos"]); plt.yticks(range(3), ["Neg", "Neu", "Pos"])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title(f"{best['embedding']} + {best['model']}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=150); plt.close()
    return results, dict(best=best, test_index=test.index, test_pred=pred)
