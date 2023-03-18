
import math
import random
import time
from collections import Counter

import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_array
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

nlp = spacy.load("en_core_web_sm")


def text_pipeline_spacy(text: str):
    tokens: list[str] = []
    doc = nlp(text)
    for t in doc:
        if not t.is_stop and not t.is_punct and not t.is_space:
            tokens.append(t.lemma_.lower())
    return tokens


def make_vocabulary(corpus: list[list[str]]):
    unique_tokens = sorted(set(t for token_list in corpus for t in token_list))
    token_to_id = {v: i for i, v in enumerate(unique_tokens)}
    return token_to_id


def make_onehot_sparse(tokens, vocab):
    sparse_onehot_vector = {}
    for token in tokens:
        sparse_onehot_vector[vocab[token]] = 1
    return sparse_onehot_vector


def doc_frequency(corpus: list[list[str]]):
    doc_freq = Counter()
    for d in corpus:
        unique_tokens = set(d)
        for t in unique_tokens:
            doc_freq[t] += 1
    return doc_freq


def make_tf_sparse(tokens, vocab):
    counts = Counter(tokens)
    sparse_vector = {vocab[t]: c for t, c in counts.items()}
    return sparse_vector


def make_tfidf_sparse(tokens, vocab, doc_freq, num_docs):
    sparse_vector = {}
    counts = Counter(tokens)
    for t, c in counts.items():
        if c > 0:
            tf = 1 + math.log10(c)
        else:
            tf = 0
        idf = math.log10(num_docs / doc_freq[t])
        sparse_vector[vocab[t]] = tf*idf
    return sparse_vector


def tfidf_vectoriser(corpus: list[str]):
    vectoriser = TfidfVectorizer(tokenizer=text_pipeline_spacy)
    return vectoriser.fit_transform(corpus)


def classifier_dummy_most_freq(dataset):
    classifier = DummyClassifier(strategy="most_frequent")

    train_data = [text_pipeline_spacy(text) for text in dataset[dataset["split"] == "train"]["tweet_body"]]
    train_labels = dataset[dataset["split"] == "train"]["class"]
    val_data = [text_pipeline_spacy(text) for text in dataset[dataset["split"] == "validation"]["tweet_body"]]
    val_labels = dataset[dataset["split"] == "validation"]["class"]

    classifier.fit(train_data, train_labels)

    train_acc = classifier.score(train_data, train_labels)
    val_acc = classifier.score(val_data, val_labels)
    train_precision, train_recall, train_f, _ = precision_recall_fscore_support(y_true=train_labels, y_pred=classifier.predict(train_data), average="macro")
    val_precision, val_recall, val_f, _ = precision_recall_fscore_support(y_true=val_labels, y_pred=classifier.predict(val_data), average="macro")

    return ((train_acc, round(train_precision, 3), round(train_recall, 3), round(train_f, 3)), (val_acc, round(val_precision, 3), round(val_recall, 3), round(val_f, 3)))


def classifier_dummy_stratified(dataset):
    classifier = DummyClassifier(strategy="stratified")

    train_data = [text_pipeline_spacy(text) for text in dataset[dataset["split"] == "train"]["tweet_body"]]
    train_labels = dataset[dataset["split"] == "train"]["class"]
    val_data = [text_pipeline_spacy(text) for text in dataset[dataset["split"] == "validation"]["tweet_body"]]
    val_labels = dataset[dataset["split"] == "validation"]["class"]

    classifier.fit(train_data, train_labels)

    train_acc = classifier.score(train_data, train_labels)
    val_acc = classifier.score(val_data, val_labels)
    train_precision, train_recall, train_f, _ = precision_recall_fscore_support(y_true=train_labels, y_pred=classifier.predict(train_data), average="macro")
    val_precision, val_recall, val_f, _ = precision_recall_fscore_support(y_true=val_labels, y_pred=classifier.predict(val_data), average="macro")

    return ((round(train_acc, 3), round(train_precision, 3), round(train_recall, 3), round(train_f, 3)), (round(val_acc, 3), round(val_precision, 3), round(val_recall, 3), round(val_f, 3)))


def classifier_logisticreg_onehot(dataset):
    classifier = LogisticRegression()

    train_corpus = [text_pipeline_spacy(text) for text in dataset[dataset["split"] == "train"]["tweet_body"]]
    val_corpus = [text_pipeline_spacy(text) for text in dataset[dataset["split"] == "validation"]["tweet_body"]]
    vocab = make_vocabulary(train_corpus)

    train_data = [make_onehot_sparse(tokens, vocab) for tokens in train_corpus]
    train_labels = dataset[dataset["split"] == "train"]["class"]
    val_data = [make_onehot_sparse(tokens, vocab) for tokens in val_corpus]
    val_labels = dataset[dataset["split"] == "validation"]["class"]

    classifier.fit(train_data, train_labels)

    train_acc = classifier.score(train_data, train_labels)
    val_acc = classifier.score(val_data, val_labels)
    train_precision, train_recall, train_f, _ = precision_recall_fscore_support(y_true=train_labels, y_pred=classifier.predict(train_data), average="macro")
    val_precision, val_recall, val_f, _ = precision_recall_fscore_support(y_true=val_labels, y_pred=classifier.predict(val_data), average="macro")

    return ((round(train_acc, 3), round(train_precision, 3), round(train_recall, 3), round(train_f, 3)), (round(val_acc, 3), round(val_precision, 3), round(val_recall, 3), round(val_f, 3)))


def classifier_logisticreg_tfidf(dataset):
    classifier = LogisticRegression()

    train_data = dataset[dataset["split"] == "train"]["tweet_body"]
    train_labels = dataset[dataset["split"] == "train"]["class"]
    val_data = dataset[dataset["split"] == "validation"]["tweet_body"]
    val_labels = dataset[dataset["split"] == "validation"]["class"]

    classifier.fit(train_data, train_labels)

    train_acc = classifier.score(train_data, train_labels)
    val_acc = classifier.score(val_data, val_labels)
    train_precision, train_recall, train_f, _ = precision_recall_fscore_support(y_true=train_labels, y_pred=classifier.predict(train_data), average="macro")
    val_precision, val_recall, val_f, _ = precision_recall_fscore_support(y_true=val_labels, y_pred=classifier.predict(val_data), average="macro")

    return ((round(train_acc, 3), round(train_precision, 3), round(train_recall, 3), round(train_f, 3)), (round(val_acc, 3), round(val_precision, 3), round(val_recall, 3), round(val_f, 3)))


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

    dataset = pd.read_csv("dataset.csv")

    # Vectorise text
    # vectoriser = TfidfVectorizer(tokenizer=text_pipeline_spacy)
    # tfidf_sparse_matrix = vectoriser.fit_transform(dataset["tweet_body"])
    # vocab = list(vectoriser.get_feature_names_out())

    print("Metrics for dummy most_frequent", classifier_dummy_most_freq(dataset=dataset))
    print("Metrics for dummy stratified", classifier_dummy_stratified(dataset=dataset))
    print("Metrics for logistic regression one-hot", classifier_logisticreg_onehot(dataset=dataset))
    # print("Metrics for logistic regression tf-idf", classifier_logisticreg_tfidf(dataset=dataset))


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start}")
