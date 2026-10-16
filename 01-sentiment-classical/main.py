import html
import re
import time

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATASET = "tblard/allocine"
SPLITS = ["train", "validation", "test"]
LABELS = ["negative", "positive"]
URL_PATTERN = r"https?://\S+|www\.\S+"
# un <[^>]+> générique supprimerait aussi les citations écrites "<< ... >"
HTML_PATTERN = r"</?(?:a|b|i|u|p|br|em|strong|span|div)\b[^<>]*>"
# le pattern par défaut enlève "!", "?", les smileys comme ":)" et les mots d'une
# lettre comme le "n" de "n'est"
TOKEN_PATTERN = r"(?u)\b\w+\b|[!?]|[:;]-?[()]"
MIN_DF = 2
NGRAM_RANGES = [(1, 1), (1, 2)]
NB_ALPHAS = [0.1, 0.3, 1.0, 3.0]
LR_CS = [1, 3, 10, 30]
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
N_ERRORS = 5
N_TOP_FEATURES = 20
SEED = 42


def load_split(split):
    path = hf_hub_download(
        DATASET, f"allocine/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pd.read_parquet(path)


def normalize(text):
    text = html.unescape(text).lower()
    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(HTML_PATTERN, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(df):
    df = df.dropna()
    df["text"] = df["review"].map(normalize)
    # comparer les textes sans ponctuation ni espaces attrape aussi les quasi-doublons
    # comme "super film!!" et "super film!!!"
    df["key"] = df["text"].str.replace(r"[\W_]+", "", regex=True)
    df = df[df["key"] != ""].drop_duplicates(subset=["key", "label"])
    # une clé encore en double ici a deux labels différents
    return df.drop_duplicates(subset="key", keep=False)


def load_data():
    raw = {split: load_split(split) for split in SPLITS}
    reviews = pd.concat(raw.values())["review"]
    n_urls = reviews.str.contains(URL_PATTERN).sum()
    n_html = reviews.str.contains(HTML_PATTERN).sum()
    print(f"raw reviews with a url: {n_urls}, with html tags: {n_html}")

    data = {split: prepare(df) for split, df in raw.items()}
    summary = pd.DataFrame(
        {
            "raw": [len(raw[split]) for split in SPLITS],
            "deduplicated": [len(data[split]) for split in SPLITS],
        },
        index=SPLITS,
    )

    seen = set(data["train"]["key"])
    for split in ["validation", "test"]:
        data[split] = data[split][~data[split]["key"].isin(seen)]
        # test est aussi nettoyé des reviews de validation, qui sert à choisir le modèle
        seen |= set(data[split]["key"])

    summary["kept"] = [len(data[split]) for split in SPLITS]
    summary["positive share"] = [
        round(data[split]["label"].mean(), 4) for split in SPLITS
    ]
    print(summary.to_string())
    return data["train"], data["validation"], data["test"]


def build_models():
    counts = CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=MIN_DF)
    tfidf = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, min_df=MIN_DF, sublinear_tf=True
    )
    return {
        "counts + naive bayes": (counts, MultinomialNB(), "alpha", NB_ALPHAS),
        "tf-idf + logistic regression": (
            tfidf,
            LogisticRegression(solver="liblinear", random_state=SEED),
            "C",
            LR_CS,
        ),
    }


def tune(vectorizer, classifier, param, values, train, val):
    best = {"val macro f1": -1.0}
    # la vectorisation avec bigrammes est la partie lente, donc on la fait une fois par
    # n-gram range et sa matrice est réutilisée pour chaque alpha ou C
    for ngram_range in NGRAM_RANGES:
        vec = clone(vectorizer).set_params(ngram_range=ngram_range)
        start = time.perf_counter()
        X_train = vec.fit_transform(train["text"])
        vec_time = time.perf_counter() - start
        X_val = vec.transform(val["text"])
        print(
            f"ngram_range={ngram_range}: {X_train.shape[1]} features, "
            f"vectorized in {vec_time:.1f} s"
        )

        for value in values:
            clf = clone(classifier).set_params(**{param: value})
            start = time.perf_counter()
            clf.fit(X_train, train["label"])
            fit_time = vec_time + time.perf_counter() - start
            score = f1_score(val["label"], clf.predict(X_val), average="macro")
            print(f"  {param}={value}: val macro f1 {score:.4f}")
            if score > best["val macro f1"]:
                best = {
                    "val macro f1": score,
                    "setting": f"ngram_range={ngram_range}, {param}={value}",
                    "pipeline": make_pipeline(vec, clf),
                    "fit s": fit_time,
                }

    print(f"best: {best['setting']}")
    return best


def tune_threshold(pipeline, val):
    proba = pipeline.predict_proba(val["text"])[:, 1]
    scores = []
    for threshold in THRESHOLDS:
        y_pred = (proba >= threshold).astype(int)
        scores.append(f1_score(val["label"], y_pred, average="macro"))
        print(f"  threshold={threshold}: val macro f1 {scores[-1]:.4f}")
    best = int(np.argmax(scores))
    print(f"best threshold: {THRESHOLDS[best]}")
    return THRESHOLDS[best], scores[best]


def evaluate(name, model, test, threshold):
    start = time.perf_counter()
    proba = model.predict_proba(test["text"])[:, 1]
    predict_time = time.perf_counter() - start

    y_true, y_pred = test["label"], (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=[f"true {label}" for label in LABELS],
        columns=[f"pred {label}" for label in LABELS],
    )
    print(f"\n{name}, test confusion matrix")
    print(cm.to_string())
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision neg": precision[0],
        "recall neg": recall[0],
        "f1 neg": f1[0],
        "precision pos": precision[1],
        "recall pos": recall[1],
        "f1 pos": f1[1],
        "macro f1": f1.mean(),
        "predict s": predict_time,
    }
    return proba, y_pred, metrics


def show_errors(test, proba, y_pred, n):
    errors = test.assign(pred=y_pred, prob=np.where(y_pred == 1, proba, 1 - proba))
    errors = errors[errors["pred"] != errors["label"]]
    print(f"{len(errors)} errors out of {len(test)}")
    for i, row in enumerate(errors.nlargest(n, "prob").itertuples(), 1):
        true, predicted = LABELS[row.label], LABELS[row.pred]
        print(f"\n{i}. true {true}, predicted {predicted} (p={row.prob:.4f})")
        print(row.review)


def show_top_features(pipeline, n):
    vectorizer, classifier = pipeline[0], pipeline[-1]
    names = vectorizer.get_feature_names_out()
    order = np.argsort(classifier.coef_[0])
    print("most negative:", ", ".join(names[order[:n]]))
    print("most positive:", ", ".join(names[order[::-1][:n]]))


def main():
    train, val, test = load_data()

    results, predictions, pipelines = {}, {}, {}
    for name, (vectorizer, classifier, param, values) in build_models().items():
        print(f"\n{name}")
        best = tune(vectorizer, classifier, param, values, train, val)
        threshold, val_f1 = tune_threshold(best["pipeline"], val)
        proba, y_pred, metrics = evaluate(name, best["pipeline"], test, threshold)
        pipelines[name] = best["pipeline"]
        predictions[name] = (proba, y_pred)
        results[name] = {
            "val macro f1": val_f1,
            "threshold": threshold,
            **metrics,
            "fit s": best["fit s"],
        }

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train["text"], train["label"])
    val_f1 = f1_score(val["label"], baseline.predict(val["text"]), average="macro")
    _, _, metrics = evaluate("majority class", baseline, test, 0.5)
    results["majority class"] = {"val macro f1": val_f1, **metrics}

    print("\ncomparison on test")
    print(pd.DataFrame(results).round(4).to_string())

    wrong = [y_pred != test["label"] for _, y_pred in predictions.values()]
    print(f"\ntest reviews misclassified by both models: {(wrong[0] & wrong[1]).sum()}")

    for name, (proba, y_pred) in predictions.items():
        print(f"\nmost confident errors, {name}")
        show_errors(test, proba, y_pred, N_ERRORS)

    print("\nlargest logistic regression weights")
    show_top_features(pipelines["tf-idf + logistic regression"], N_TOP_FEATURES)


if __name__ == "__main__":
    main()
