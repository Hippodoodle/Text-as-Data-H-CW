
import math
import random
from collections import Counter

import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_array

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


def pick_random_centroids(num_centroids, vocab_size, seed=42):
    centroids = []
    random_generator = random.Random(seed)
    for i in range(num_centroids):
        centroid = {}
        for j in range(vocab_size):
            centroid[j] = random_generator.random()
        centroids.append(centroid)
    return centroids


def sparse_dot_prod(sv1, sv2):
    dot_prod = 0
    indices = set(sv1).intersection(sv2)
    for i in indices:
        dot_prod += sv1.get(i, 0) * sv2.get(i, 0)
    return dot_prod


def sparse_cosine_similarity(sv1, sv2):
    d1 = math.sqrt(sum(val*val for index, val in sv1.items()))
    d2 = math.sqrt(sum(val*val for index, val in sv2.items()))
    return sparse_dot_prod(sv1, sv2) / (d1*d2)


def assign_to_cluster(vector, centroids):
    best_centroid_index = 0
    best_score = 0
    for i in range(len(centroids)):
        score = sparse_cosine_similarity(vector, centroids[i])
        if score > best_score:
            best_score = score
            best_centroid_index = i
    return best_centroid_index


def main():
    """
    class:
        0 -> hate speech
        1 -> offensive language
        2 -> neither

    tweet_id,class,tweet_body,split
    0,2,"!!!...",training
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

    # print(text_pipeline_spacy(dataset["tweet_body"][0]))
    # documents: list[dict] = list(dataset.to_dict(orient="index").values())

    """
    corpus = []
    for doc in tqdm(dataset["tweet_body"]):
        corpus.append(text_pipeline_spacy(doc))
    print(corpus[:5])
    """

    """
    for doc in tqdm(documents):
        doc['tokens'] = text_pipeline_spacy(doc["tweet_body"])

    documents_tokens = [doc['tokens'] for doc in documents]
    documents_vocab = make_vocabulary(documents_tokens)
    documents_docfreq = doc_frequency(documents_tokens)
    num_docs = len(documents_tokens)

    documents_sparse = [make_tf_sparse(doc['tokens'], documents_vocab) for doc in documents]
    documents_tfidf_vectors = [make_tfidf_sparse(doc, documents_vocab, documents_docfreq, num_docs) for doc in documents_tokens]

    print(documents_sparse[:1])
    print(documents_tfidf_vectors[:1])
    """

    # Step 0: Vectorise text
    vectoriser = TfidfVectorizer(tokenizer=text_pipeline_spacy)
    tfidf_sparse_matrix = vectoriser.fit_transform(dataset["tweet_body"])
    vocab = list(vectoriser.get_feature_names_out())
    # print(tfidf_sparse_matrix, tfidf_sparse_matrix.shape, type(tfidf_sparse_matrix))
    # print(vocab[14501], vocab[11115], vocab[5367], vocab[8961], vocab[6643], vocab[15184], vocab[3283], vocab[13065])

    # Step 1: Pick k random "centroids"
    centroids = pick_random_centroids(5, len(vocab))
    print(len(centroids), len(centroids[0]))

    # Step 2: Assign each vector to its closest centroid
    cluster_ids = []
    vectors = []
    for scipy_vector in tfidf_sparse_matrix:
        vector = dict(zip(scipy_vector.indices, scipy_vector.data))
        vectors.append(vector)
        cluster_ids.append(assign_to_cluster(vector, centroids))
    print(cluster_ids[:10])

    # Step 3: Recalculate the centroids based on the closest vectors
    centroids = []
    unique_clusters = set(cluster_ids)

    for cluster_id in unique_clusters:
        cluster_vectors = []
        for i in range(len(vectors)):
            if cluster_ids[i] == cluster_id:
                cluster_vectors.append(vectors[i])

        common_keys = set()  # TODO: understand this bit
        for v in cluster_vectors:
            common_keys.update(v.keys())

        centroid = {}
        for key in common_keys:
            values = []
            for v in cluster_vectors:
                values.append(v.get(key, 0))
            avg_value = sum(values) / len(cluster_vectors)
            centroid[key] = avg_value

        centroids.append(centroid)

    print(len(centroids), len(centroids[0]))

    """
    centroids = []
    clusters = set(cluster_ids)
    for cluster_id in clusters:
    """


if __name__ == "__main__":
    main()
