import math
from collections import defaultdict
from typing import List
import nltk
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
from sentence_transformers import CrossEncoder
from abc import ABC, abstractmethod
import numpy as np


class DocsRanker(ABC):
    @abstractmethod
    def rankDocuments(self, query, docs):
        pass


class BM25WithProximity:
    def __init__(self, documents: List[List[str]], k1=1.5, b=0.75, proximity_weight=0.6):
        self.k1 = k1
        self.b = b
        self.proximity_weight = proximity_weight
        self.documents = documents
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.total_docs = len(self.documents)

        self.doc_freqs = defaultdict(int)
        for doc in self.documents:
            for term in set(doc):
                self.doc_freqs[term] += 1

    def _bm25_term_score(self, term, doc, doc_length):
        term_freq = doc.count(term)
        if term_freq == 0:
            return 0
        idf = math.log((self.total_docs) /
                       (self.doc_freqs[term] + 1.0) + 1)
        normalization = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
        score = idf * ((term_freq * (self.k1 + 1)) /
                       (term_freq + self.k1 * normalization))
        return score

    def _proximity_score(self, query_terms, doc):
        positions = {term: [] for term in query_terms}
        for index, word in enumerate(doc):
            if word in query_terms:
                positions[word].append(index)


        min_distance = float('inf')
        for i, term1 in enumerate(query_terms):
            for term2 in query_terms[i + 1:]:
                for pos1 in positions[term1]:
                    for pos2 in positions[term2]:
                        distance = abs(pos1 - pos2)
                        if distance < min_distance:
                            min_distance = distance


        if min_distance == float('inf'):
            return 0
        return 1 / (1 + min_distance)

    def score(self, query: List[str], doc_index: int):
        doc = self.documents[doc_index]
        doc_length = self.doc_lengths[doc_index]

        bm25_score = sum(self._bm25_term_score(term, doc, doc_length) for term in query)

        proximity_score = self._proximity_score(query, doc)

        total_score = bm25_score + self.proximity_weight * proximity_score
        return total_score

    def get_scores(self, query: List[str]):
        scores = [self.score(query, doc_index) for doc_index in range(self.total_docs)]
        return scores

class Bm25Ranker(DocsRanker):
    # preprocess_func: переобразует запрос и документ в список слов
    def __init__(self, bm25_alg = BM25WithProximity, preprocess_func = None) -> None:
        self.preprocess_func = preprocess_func
        self.bm25_alg = bm25_alg

    def rankDocuments(self, query, docs):
        if self.preprocess_func is None:
            self.preprocess_func = lambda doc: doc.split()
        tokenized_corpus = [doc.split() for doc in docs]
        bm25 = self.bm25_alg(tokenized_corpus)
        tokenized_query = self.preprocess_func(query)
        return bm25.get_scores(tokenized_query)

def lemmatize(doc):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(word)[0].normal_form for word in doc.split()]

def stem(doc):
   stemmer = SnowballStemmer("russian")
   return [stemmer.stem(word) for word in doc.split()]


class CrossEncoderRanker(DocsRanker):
    def __init__(self) -> None:
        # self.reranker_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device='cuda')
        self.reranker_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device='cpu')

    def rankDocuments(self, query, docs):
        return np.array([self.reranker_model.predict([[query, doc]])[0] for doc in docs])
    

def calculate_relevance(query: str, document: str, preprocess_func = None) -> float:
    """
    Оценивает релевантность документа запросу на основе кворума (числа совпавших слов).
    
    :param query: текст запроса
    :param document: текст документа
    :return: значение релевантности от 0 до 1
    """

    if preprocess_func is None:
        preprocess_func = lambda doc: doc.split()

    query_words = set(preprocess_func(query))
    document_words = set(document.split())

    if len(query_words) == 0:
        return 0.0
    
    intersection = query_words.intersection(document_words)
    relevance = len(intersection) / len(query_words)
    return relevance

def documents_filter_quorum(query: str, documents: List[str], preprocess_func = None, threshold: float = 0.5) -> List[str]:
    return [doc for doc in documents if calculate_relevance(query, doc, preprocess_func) >= threshold]
