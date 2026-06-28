import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    
    vocab = sorted(
        set(
            word.lower()
            for doc in documents
            for word in doc.split()
        )
    )

    N = len(documents)

    idf = {}

    for word in vocab:
        df_t = sum(
            1
            for doc in documents
            if word in doc.lower().split()
        )

        idf[word] = np.log(N / df_t)

    tfidf_matrix = []

    for doc in documents:

        t = Counter(word.lower() for word in doc.split())
        t_d = len(doc.split())

        row = []

        for word in vocab:
            tf = t[word] / t_d
            row.append(tf * idf[word])

        tfidf_matrix.append(row)

    return np.array(tfidf_matrix), vocab