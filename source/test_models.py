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
from rankers.rankers import CrossEncoderRanker, Bm25Ranker, BM25WithProximity, stem, documents_filter_quorum, \
    BiEncoderRanker, QuorumRanker
from poiskovik import Poiskovik

# Настройка логирования
logging.basicConfig(
    filename='logs/queries.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

class PoiskovikTest(Poiskovik):
    def __init__(self, request, client_address, server):
        self.use_stemming = True
        self.quorum_threshold = -1
        self.partForRanker2 = 0.1
        self.ranker2 = None
        self.data_base_type = "monolit"
        self.isItTest = True
        self.noSummarize = True
        #super().__init__(request, client_address, server)
        #self.index = self.getVectorDB(self.vectorDBPath)
        #self.use_stemming = True
        # self.ranker = Bm25Ranker(bm25_alg = BM25WithProximity, preprocess_func = stem if self.use_stemming else None)
        #self.ranker = CrossEncoderRanker()

    def test(self, filename, ranker = None, kDocuments = 50):
        self.ranker = ranker
        self.kDocs = kDocuments

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
        pos_arr_light = -np.ones(len(queries))

        for i in range(len(queries)):
            urls = self.processQuery(queries[i])

            for j in range(len(urls)):
                if urls[j].split('#')[0] == real_urls[i].split('#')[0] and pos_arr_light[i] == -1:
                    pos_arr_light[i] = j
                if urls[j] == real_urls[i]:
                    pos_arr[i] = j
                    break
        return pos_arr, pos_arr_light

def metric_inv(n, coef = 5):
    if n == -1:
        return 0
    else:
        return coef / (n + coef)

def metric_in_top_5(n):
    return 1 if -1 < n < 5 else 0

def metric_in_top_15(n):
    return 1 if -1 < n < 15 else 0

def metric_in_top_30(n):
    return 1 if -1 < n < 30 else 0

def general_test(filenames, rankers, rankers2, treshholds, to_rerank_fracs, document_nums, metrics):
    res_str = ''
    tester = PoiskovikTest(None, None, None)
    for ranker2 in rankers2:
        tester.ranker2 = ranker2[1]
        for treshhold in treshholds:
            tester.quorum_threshold = treshhold
            for to_rerank_frac in to_rerank_fracs:
                tester.partForRanker2 = to_rerank_frac
                for ranker in rankers:
                    for kDocuments in document_nums:
                        scoresStrongMatches = np.zeros((len(filenames), len(metrics)))
                        scoresLightMatches = np.zeros((len(filenames), len(metrics)))
                        for k in range(len(filenames)):
                            filename = filenames[k]
                            posOfMatches, posOfLightMatches = tester.test(filename[1], ranker = ranker[1], kDocuments = kDocuments)
                            for i in range(len(metrics)):
                                scoreStrong = np.zeros(posOfMatches.shape[0])
                                scoreLight = np.zeros(posOfLightMatches.shape[0])
                                for j in range(posOfMatches.shape[0]):
                                    scoreStrong[j] = metrics[i][1](posOfMatches[j])
                                    scoreLight[j] = metrics[i][1](posOfLightMatches[j])
                                scoresStrongMatches[k][i] = scoreStrong.mean()
                                scoresLightMatches[k][i] = scoreLight.mean()
                        res_str += f'ranker: {ranker[0]}, ranker2: {ranker2[0]}, treshold: {treshhold}, part for r2: {to_rerank_frac}, kDocuments: {kDocuments}\n'
                        scoreStrongMean = scoresStrongMatches.mean(axis = 0)
                        scoreLightMean = scoresLightMatches.mean(axis = 0)

                        res_str += 'Strong condition metrics. '
                        for i in range(len(metrics)):
                            res_str += f' {metrics[i][0]}: {scoreStrongMean[i]}'

                        res_str += '\nLight condition metrics. '
                        for i in range(len(metrics)):
                            res_str += f' {metrics[i][0]}: {scoreLightMean[i]}'
                        print(res_str)
                        res_str = ''
    tester.sqlConnectionMonolit.close()

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        filenames = []
        rankers = []
        rankers2 = []
        treshholds = []
        to_rerank_fracs = []
        document_nums = []
        metrics = []
        with open(argv[1], encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line)
            for word in lines[0][:-1].split(' '):
                if word == '#' or word == '//':
                    break
                filenames.append([word, 'text_parser/data/queries_split/' + word + '.txt'])
            for word in lines[1][:-1].split(' '):
                if word == 'None':
                    rankers.append([word, None])
                if word == 'BM25':
                    rankers.append([word, Bm25Ranker(bm25_alg = BM25WithProximity, preprocess_func = stem)])
                if word == 'CrossEncoder':
                    rankers.append([word, CrossEncoderRanker()])
                if word == 'BiEncoder':
                    rankers.append([word, BiEncoderRanker()])
                if word == 'Quorum':
                    rankers.append([word, QuorumRanker()])
                if word == '#' or word == '//':
                    break
            for word in lines[2][:-1].split(' '):
                if word == 'None':
                    rankers2.append([word, None])
                if word == 'BM25':
                    rankers2.append([word, Bm25Ranker(bm25_alg = BM25WithProximity, preprocess_func = stem)])
                if word == 'CrossEncoder':
                    rankers2.append([word, CrossEncoderRanker()])
                if word == 'BiEncoder':
                    rankers2.append([word, BiEncoderRanker()])
                if word == 'Quorum':
                    rankers2.append([word, QuorumRanker()])
                if word == '#' or word == '//':
                    break
            for word in lines[3][:-1].split(' '):
                if word == '#' or word == '//':
                    break
                treshholds.append(float(word))
            for word in lines[4][:-1].split(' '):
                if word == '#' or word == '//':
                    break
                to_rerank_fracs.append(float(word))
            for word in lines[5][:-1].split(' '):
                if word == '#' or word == '//':
                    break
                document_nums.append(int(word))
            for word in lines[6][:-1].split(' '):
                if word == 'inv_5':
                    metrics.append([word, metric_inv])
                if word == 'in_top_5':
                    metrics.append([word, metric_in_top_5])
                if word == 'in_top_15':
                    metrics.append([word, metric_in_top_15])
                if word == 'in_top_30':
                    metrics.append([word, metric_in_top_30])
                if word == '#' or word == '//':
                    break

        general_test(filenames, rankers, rankers2, treshholds, to_rerank_fracs, document_nums, metrics)
