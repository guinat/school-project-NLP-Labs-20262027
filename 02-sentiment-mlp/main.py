import copy
import html
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent / "results"
DATASET = "tblard/allocine"
SPLITS = ["train", "validation", "test"]
LABELS = ["negative", "positive"]
URL_PATTERN = r"https?://\S+|www\.\S+"
# un <[^>]+> générique supprimerait aussi les citations écrites avec "<< ... >"
HTML_PATTERN = r"</?(?:a|b|i|u|p|br|em|strong|span|div)\b[^<>]*>"
# le pattern par défaut enlève "!", "?", les smileys comme ":)" et les mots d'une
# lettre comme le "n" de "n'est"
TOKEN_PATTERN = r"(?u)\b\w+\b|[!?]|[:;]-?[()]"
MIN_DF = 2

# meilleur réglage tf-idf + logreg du TP 1 (choisi sur le macro F1 de validation);
# comme au TP 1, le seuil de décision de chaque modèle est ensuite réglé sur validation
NGRAM_RANGE = (1, 2)
LOGREG_C = 10
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

MLP_MAX_FEATURES = 20_000
SVD_COMPONENTS = 256
HIDDEN_SIZE = 128
DROPOUT = 0.3
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 4096
MAX_EPOCHS = 20
PATIENCE = 3
LEARNING_RATES = [1e-4, 1e-3, 1e-2, 1e-1]
N_ERRORS = 5
LOSS_TICKS = [0.15, 0.2, 0.3, 0.5, 1.0]
VAL_LOSS_ZOOM = (0.2, 0.26)


def normalize(text):
    text = html.unescape(text).lower()
    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(HTML_PATTERN, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_split(split):
    path = hf_hub_download(
        DATASET, f"allocine/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path).dropna()
    df["text"] = df["review"].map(normalize)
    # comparer les textes sans ponctuation ni espaces attrape aussi les quasi-doublons
    # comme "super film!!" et "super film !!!"
    df["key"] = df["text"].str.replace(r"[\W_]+", "", regex=True)
    df = df[df["key"] != ""].drop_duplicates(subset=["key", "label"])
    # une clé encore en double ici a deux labels différents
    return df.drop_duplicates(subset="key", keep=False)


def load_data():
    data = {split: load_split(split) for split in SPLITS}
    seen = set(data["train"]["key"])
    for split in ["validation", "test"]:
        data[split] = data[split][~data[split]["key"].isin(seen)]
        seen |= set(data[split]["key"])
    for split, df in data.items():
        print(f"{split}: {len(df)} reviews, {df['label'].mean():.1%} positive")
    return [data[split].reset_index(drop=True) for split in SPLITS]


def make_tfidf(max_features=None):
    return TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=max_features,
        sublinear_tf=True,
    )


def make_logreg():
    return LogisticRegression(C=LOGREG_C, solver="liblinear", random_state=42)


def make_mlp(n_inputs):
    return nn.Sequential(
        nn.Linear(n_inputs, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_SIZE, 2),
    )


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def to_tensor(X):
    # les lignes tf-idf creuses sont passées en dense un mini-batch à la fois
    if sp.issparse(X):
        X = X.toarray()
    return torch.tensor(X, dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, X):
    model.eval()
    return torch.cat(
        [
            model(to_tensor(X[i : i + EVAL_BATCH_SIZE]))
            for i in range(0, X.shape[0], EVAL_BATCH_SIZE)
        ]
    )


def mlp_proba(model, X):
    return predict_logits(model, X).softmax(dim=1)[:, 1].numpy()


def train_mlp(X_train, y_train, X_val, y_val, lr):
    torch.manual_seed(1337)
    model = make_mlp(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_train_t, y_val_t = torch.tensor(y_train), torch.tensor(y_val)
    history = {"train": [], "val": []}
    best_loss, best_epoch, best_state = float("inf"), 0, None
    start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(y_train_t)).split(BATCH_SIZE):
            logits = model(to_tensor(X_train[idx.numpy()]))
            loss = F.cross_entropy(logits, y_train_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # les deux loss sont calculées en mode eval (sans dropout) pour les comparer
        train_loss = F.cross_entropy(predict_logits(model, X_train), y_train_t).item()
        val_loss = F.cross_entropy(predict_logits(model, X_val), y_val_t).item()
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch, time.perf_counter() - start


def learning_rate_study(X_train, y_train, X_val, y_val):
    runs, rows = {}, []
    for lr in LEARNING_RATES:
        print(f"\nsvd + mlp, lr={lr:g}")
        model, history, best_epoch, train_time = train_mlp(
            X_train, y_train, X_val, y_val, lr
        )
        runs[lr] = (model, history, best_epoch, train_time)
        val_pred = mlp_proba(model, X_val) >= 0.5
        rows.append(
            {
                "lr": lr,
                "epochs": len(history["val"]),
                "best epoch": best_epoch,
                "train loss": history["train"][best_epoch - 1],
                "val loss": history["val"][best_epoch - 1],
                "val macro f1": f1_score(y_val, val_pred, average="macro"),
                "time s": train_time,
            }
        )
    return runs, pd.DataFrame(rows)


def fit_and_predict(
    model, X_train, y_train, X_val, X_test, prep_time=0.0, prep_test_time=0.0
):
    _, train_time = timed(model.fit, X_train, y_train)
    test_proba, predict_time = timed(model.predict_proba, X_test)
    val_proba = model.predict_proba(X_val)[:, 1]
    return (
        val_proba,
        test_proba[:, 1],
        prep_time + train_time,
        prep_test_time + predict_time,
    )


def predict_mlp(model, X_val, X_test, train_time, prep_test_time):
    test_proba, predict_time = timed(mlp_proba, model, X_test)
    return (
        mlp_proba(model, X_val),
        test_proba,
        train_time,
        prep_test_time + predict_time,
    )


def tune_threshold(y_val, val_proba):
    scores = [f1_score(y_val, val_proba >= t, average="macro") for t in THRESHOLDS]
    best = int(np.argmax(scores))
    return THRESHOLDS[best], scores[best]


def score_on_test(name, y_true, y_pred):
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=[f"true {label}" for label in LABELS],
        columns=[f"pred {label}" for label in LABELS],
    )
    print(f"\n{name}, test confusion matrix")
    print(cm.to_string())
    f1_neg, f1_pos = f1_score(y_true, y_pred, average=None)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1 neg": f1_neg,
        "f1 pos": f1_pos,
        "macro f1": (f1_neg + f1_pos) / 2,
    }


def compare_on_test(models, y_val, y_test):
    rows, preds = {}, {}
    for name, (val_proba, test_proba, train_time, predict_time) in models.items():
        threshold, val_f1 = tune_threshold(y_val, val_proba)
        preds[name] = (test_proba >= threshold).astype(int)
        rows[name] = {
            "threshold": threshold,
            "val macro f1": val_f1,
            **score_on_test(name, y_test, preds[name]),
            "train s": train_time,
            "predict s": predict_time,
        }
    return pd.DataFrame.from_dict(rows, orient="index"), preds


def plot_loss_curves(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(11, 4))
    for ax, (name, (history, best_epoch)) in zip(axes, runs.items(), strict=True):
        epochs = range(1, len(history["train"]) + 1)
        ax.plot(epochs, history["train"], marker="o", label="train")
        ax.plot(epochs, history["val"], marker="o", label="validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", label="best epoch")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=120)
    plt.close(fig)


def plot_learning_rates(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for lr, history in histories.items():
        epochs = range(1, len(history["train"]) + 1)
        for ax, split in zip(axes, ["train", "val", "val"], strict=True):
            ax.plot(epochs, history[split], marker="o", label=f"lr={lr:g}")
    titles = ["train loss", "validation loss", "validation loss, zoom"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    # lr=0.1 est loin au-dessus des autres, une échelle log le garde sur les mêmes axes
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_yticks(LOSS_TICKS, labels=[f"{t:g}" for t in LOSS_TICKS])
        ax.minorticks_off()
    axes[2].set_ylim(*VAL_LOSS_ZOOM)
    axes[0].set_ylabel("cross-entropy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_rates.png", dpi=120)
    plt.close(fig)


def show_errors(test, preds, probas, name):
    y_test = test["label"].to_numpy()
    wrong = np.flatnonzero(preds[name] != y_test)
    rng = np.random.default_rng(42)
    print(f"\n{N_ERRORS} random errors of {name} (p = probability of positive)")
    for i in rng.choice(wrong, N_ERRORS, replace=False):
        scores = ", ".join(
            f"{model} p={proba[i]:.3f}" for model, proba in probas.items()
        )
        print(f"\ntest #{i}, true {LABELS[y_test[i]]}: {scores}")
        print(test.loc[i, "review"])


def main():
    start = time.perf_counter()
    RESULTS_DIR.mkdir(exist_ok=True)
    train, val, test = load_data()
    y_train, y_val, y_test = (df["label"].to_numpy() for df in [train, val, test])
    models = {}

    print("\nfitting tf-idf + logistic regression (lab 1 model)")
    logreg = make_pipeline(make_tfidf(), make_logreg())
    models["tf-idf + logreg"] = fit_and_predict(
        logreg, train["text"], y_train, val["text"], test["text"]
    )
    print(f"features: {len(logreg[0].vocabulary_)}")

    tfidf = make_tfidf(MLP_MAX_FEATURES)
    X_train, tfidf_time = timed(tfidf.fit_transform, train["text"])
    X_val = tfidf.transform(val["text"])
    X_test, tfidf_test_time = timed(tfidf.transform, test["text"])
    X_train, X_val, X_test = (X.astype(np.float32) for X in [X_train, X_val, X_test])

    svd = make_pipeline(
        TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42), StandardScaler()
    )
    Z_train, svd_time = timed(svd.fit_transform, X_train)
    Z_val = svd.transform(X_val)
    Z_test, svd_test_time = timed(svd.transform, X_test)
    svd_time += tfidf_time
    svd_test_time += tfidf_test_time
    explained = svd[0].explained_variance_ratio_.sum()
    print(
        f"reduced input: {MLP_MAX_FEATURES} tf-idf features, then "
        f"{SVD_COMPONENTS} svd components ({explained:.1%} of the variance)"
    )

    models["tf-idf 20k + logreg"] = fit_and_predict(
        make_logreg(), X_train, y_train, X_val, X_test, tfidf_time, tfidf_test_time
    )
    models["svd + logreg"] = fit_and_predict(
        make_logreg(), Z_train, y_train, Z_val, Z_test, svd_time, svd_test_time
    )

    for n_inputs in [MLP_MAX_FEATURES, SVD_COMPONENTS]:
        n_params = sum(p.numel() for p in make_mlp(n_inputs).parameters())
        print(f"mlp: {n_inputs} -> {HIDDEN_SIZE} -> 2, {n_params} parameters")

    runs, lr_table = learning_rate_study(Z_train, y_train, Z_val, y_val)
    print()
    print(lr_table.round(4).to_string(index=False))
    print(f"learning rate study: {lr_table['time s'].sum():.0f}s")
    best_lr = lr_table.loc[lr_table["val macro f1"].idxmax(), "lr"]
    print(f"selected lr: {best_lr:g}")
    plot_learning_rates({lr: history for lr, (_, history, _, _) in runs.items()})

    print(f"\ntf-idf 20k + mlp, lr={best_lr:g}")
    mlp, history, best_epoch, mlp_time = train_mlp(
        X_train, y_train, X_val, y_val, best_lr
    )
    models["tf-idf 20k + mlp"] = predict_mlp(
        mlp, X_val, X_test, tfidf_time + mlp_time, tfidf_test_time
    )
    svd_mlp, svd_history, svd_best_epoch, svd_mlp_time = runs[best_lr]
    models["svd + mlp"] = predict_mlp(
        svd_mlp, Z_val, Z_test, svd_time + svd_mlp_time, svd_test_time
    )
    plot_loss_curves(
        {
            f"tf-idf 20k + mlp (lr={best_lr:g})": (history, best_epoch),
            f"svd + mlp (lr={best_lr:g})": (svd_history, svd_best_epoch),
        }
    )

    table, preds = compare_on_test(models, y_val, y_test)
    print("\ncomparison on test")
    print(table.round(4).to_string())

    best_mlp = table.loc[["tf-idf 20k + mlp", "svd + mlp"], "val macro f1"].idxmax()
    print(f"\nmlp with the best validation macro f1: {best_mlp}")
    mlp_wrong = preds[best_mlp] != y_test
    logreg_wrong = preds["tf-idf + logreg"] != y_test
    print(f"both wrong: {(mlp_wrong & logreg_wrong).sum()}")
    print(f"only {best_mlp} wrong: {(mlp_wrong & ~logreg_wrong).sum()}")
    print(f"only tf-idf + logreg wrong: {(~mlp_wrong & logreg_wrong).sum()}")

    probas = {name: models[name][1] for name in [best_mlp, "tf-idf + logreg"]}
    for name in probas:
        show_errors(test, preds, probas, name)

    print(f"\ntotal time: {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
