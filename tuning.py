
import time
import warnings

import pandas as pd
import spacy
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid

nlp = spacy.load("en_core_web_sm")


def text_pipeline_spacy(text: str):
    tokens: list[str] = []
    doc = nlp(text)
    for t in doc:
        if not t.is_stop and not t.is_punct and not t.is_space:
            tokens.append(t.lemma_.lower())
    return tokens


def main():
    """
    class:
        0 -> hate speech
        1 -> offensive language
        2 -> neither

    tweet_id,class,tweet_body,split
    0,2,"!!!...",train
    ...

    Class Balance:
        0: 1400
        1: 2800
        2: 2800
        total: 7000

    Train/Val/Test Split:
        60/20/20
    """
    ConvergenceWarning('ignore')

    dataset = pd.read_csv("dataset.csv")

    param_grid = {'C': [.01, 1, 100, 10000, 1000000], 'max_features': [None, 5000, 10000, 25000, 50000], 'sublinear_tf': [False, True], 'max_df': [0.5, 0.7, 0.9, 1.0]}

    best_params, best_f1 = None, 0
    for params in ParameterGrid(param_grid):
        vectoriser = TfidfVectorizer(max_features=params['max_features'], sublinear_tf=params['sublinear_tf'], max_df=params['max_df'])

        train_data = vectoriser.fit_transform(dataset[dataset["split"] == "train"]["tweet_body"])
        train_labels = list(dataset[dataset["split"] == "train"]["class"])
        val_data = vectoriser.transform(dataset[dataset["split"] == "validation"]["tweet_body"])
        val_labels = list(dataset[dataset["split"] == "validation"]["class"])

        classifier = LogisticRegression(random_state=42, C=params['C'])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            classifier.fit(train_data, train_labels)

        labels_predicted = classifier.predict(val_data)

        accuracy = accuracy_score(y_true=val_labels, y_pred=labels_predicted)
        precision = precision_score(y_true=val_labels, y_pred=labels_predicted, average="macro", zero_division=0)
        recall = recall_score(y_true=val_labels, y_pred=labels_predicted, average="macro")
        f1 = f1_score(y_true=val_labels, y_pred=labels_predicted, average="macro")
        print(f"Evaluating {params=} {accuracy=:.3f} {precision=:.3f} {recall=:.3f} {f1=:.3f}")

        if f1 > best_f1:
            best_params = params
            best_f1 = f1

    print(f"best params: {best_params}")
    print(f"best f1: {round(best_f1, 3)}")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {round(end - start, 2)}s")
