from http.server import BaseHTTPRequestHandler, HTTPServer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from urllib.parse import unquote
import sqlite3
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from rankers.rankers import CrossEncoderRanker, Bm25Ranker, BM25WithProximity, stem, documents_filter_quorum
from poiskovik import Poiskovik

# Настройка логирования
logging.basicConfig(
    filename='logs/queries.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

class PoiskovikTest(Poiskovik):
    def __init__(self, request, client_address, server):
        pass
        #super().__init__(request, client_address, server)
        #self.index = self.getVectorDB(self.vectorDBPath)
        #self.use_stemming = True
        # self.ranker = Bm25Ranker(bm25_alg = BM25WithProximity, preprocess_func = stem if self.use_stemming else None)
        #self.ranker = CrossEncoderRanker()
        #self.quorum_threshold = 0.25

    def test(self, filename, ranker = None, kDocuments = 50):
        real_urls = []
        queries = []
        with open(filename, encoding='utf-8') as f:
            prev_line = ''
            for line in f:
                if (prev_line == '##\n'):
                    real_urls.append(line[:-1])
                if (prev_line == '#\n'):
                    queries.append(line[:-1])
                prev_line = line

        pos_arr = -np.ones(len(queries))
        time_arr = np.zeros(len(queries))

        #start_time = time.time()

        for i in range(len(queries)):
            query = queries[i]
            
            start_time = time.time()
            
            #indexesForQueries = self.findVectorsIndexes([query], self.modelEncoder, kDocuments)

            # Получение документов и ссылок для всех вопросов
            if self.data_base_type == "monolit":
                urlsAndDocs = self.prepareDocsAndUrlsMonolitDb(queries, kDocuments, self.sqlConnectionMonolit, self.indexDbMonolit, start_time)
            else:
                self.sqlConnectionMonolit.close()
                urlsAndDocs = self.prepareDocsAndUrlsShardedDb(queries, kDocuments, start_time)

            urls, docs, fullDocs = urlsAndDocs[0], urlsAndDocs[1], urlsAndDocs[2]
            docs = documents_filter_quorum(query, docs, stem if self.use_stemming else None, self.quorum_threshold)

            if not ranker is None and len(docs) > 0:
                doc_scores = ranker.rankDocuments(query, docs)
                sorted_idx = np.argsort(doc_scores)
                urls = list(urls.iloc[sorted_idx[::-1]])
                #resDocs, urls, scores = list(fullDocs.iloc[sorted_idx[::-1]]), list(urls.iloc[sorted_idx[::-1]]), doc_scores[sorted_idx[::-1]]
            
            end_time = time.time()

            for j in range(len(urls)):
                if urls[j] == real_urls[i]:
                    pos_arr[i] = j
                    break
            time_arr[i] = end_time - start_time
        return pos_arr, time_arr

def metric_inv(n, coef = 5):
    if n == -1:
        return 0
    else:
        return coef / (n + coef)

def metric_in_top_k(n, k = 5):
    if -1 < n < k:
        return 1.0
    else:
        return 0.0

def general_test(filenames, rankers, document_nums, metrics):
    res_str = ''
    for ranker in rankers:
        for kDocuments in document_nums:
            scores = np.zeros((len(filenames), len(metrics)))
            times = np.zeros(len(filenames))
            for k in range(len(filenames)):
                filename = filenames[k]
                tester = PoiskovikTest(None, None, None)
                p, t = tester.test(filename[1], ranker = ranker[1], kDocuments = kDocuments)
                for i in range(len(metrics)):
                    scored = np.zeros(p.shape[0])
                    for j in range(p.shape[0]):
                        scored[j] = metrics[i][1](p[j])
                    scores[k][i] = scored.mean()
                times[k] = t.mean()
            res_str += f'ranker: {ranker[0]}, kDocuments: {kDocuments}\n'
            times = times.mean()
            scores = scores.mean(axis = 0)
            res_str += f'time: {times}'
            for i in range(len(metrics)):
                res_str += f' {metrics[i][0]}: {scores[i]}'
            res_str += '\n'
    print(res_str)

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        filenames = []
        rankers = []
        document_nums = []
        metrics = []
        with open(argv[1], encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line)
            for word in lines[0][:-1].split(' '):
                filenames.append([word, 'text_parser/data/queries_split/' + word + '.txt'])
            for word in lines[1][:-1].split(' '):
                if word == 'None':
                    rankers.append([word, None])
                if word == 'BM25':
                    rankers.append([word, Bm25Ranker(bm25_alg = BM25WithProximity, preprocess_func = stem)])
                if word == 'CrossEncoder':
                    rankers.append([word, CrossEncoderRanker()])
            for word in lines[2][:-1].split(' '):
                document_nums.append(int(word))
            for word in lines[3][:-1].split(' '):
                if word == 'inv_5':
                    metrics.append([word, metric_inv])
                if word == 'in_top_5':
                    metrics.append([word, metric_in_top_k])
        general_test(filenames, rankers, document_nums, metrics)
