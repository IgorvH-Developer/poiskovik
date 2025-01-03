from http.server import BaseHTTPRequestHandler, HTTPServer
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from urllib.parse import unquote
import sqlite3
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from rankers.rankers import Bm25Ranker, BM25WithProximity, stem, documents_filter_quorum, CrossEncoderRanker, \
    BiEncoderRanker
from utils import get_rows_from_sql, findVectorsIndexes, findDocIndexesByTextSearch, clean_string
from whoosh.index import open_dir

# Настройка логирования
logging.basicConfig(
    filename='../logs/queries.log', filemode='a', level=logging.WARNING, format='%(message)s'
)


class Poiskovik(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def logDetails(self, text):
        logging.warning(f"{text}")

    def logDetails(self, text, startTime):
        endTime = time.time()
        processing_time = endTime - startTime
        logging.warning(f"{text} {processing_time:.2f}.")

    def summarizeText(self, docs, query=None):
        max_input = 512
        task_prefix = "Нади ответ на вопрос: "
        input_seq = ""
        if query != None:
            input_seq = " " + query + "\n"
        input_seq += "\n".join(docs)
        input_seq = [input_seq]
        encoded = self.tokenizer(
            [task_prefix + sequence for sequence in input_seq],
            max_length=max_input,
            truncation=True,
            return_tensors="pt",
        )
        predicts = self.modelSummarizer.generate(encoded['input_ids'])
        return self.tokenizer.batch_decode(predicts, skip_special_tokens=True)

    def splitAllQueries(self, allQueries):
        queries = []
        responses = []
        for question in allQueries.split('\n\n'):
            if question in self.queryHistory.keys():
                responses.append(self.queryHistory[question])
                continue
            queries.append(question)
        return queries, responses

    def combineVectorAndTextIndexes(self, vectIdxs, vectorScores, textIdxs, textScores):
        # 1 способ комбинирования
        # if len(vectIdxs) < len(textIdxs):
        #     combinedIdx[list(range(0, len(vectIdxs)*2, 2))] = vectIdxs
        #     combinedIdx[list(range(1, len(vectIdxs)*2, 2))] = textIdxs[:len(vectIdxs)]
        #     combinedIdx[len(vectIdxs)*2:] = textIdxs[len(vectIdxs):]
        # else:
        #     combinedIdx[list(range(0, len(textIdxs) * 2, 2))] = vectIdxs[:len(textIdxs)]
        #     combinedIdx[list(range(1, len(textIdxs) * 2 + 1, 2))] = textIdxs
        #     combinedIdx[len(textIdxs) * 2:] = vectIdxs[len(textIdxs):]
        # return combinedIdx

        # 2 способ комбинирования
        textScoresNorm = []
        if len(textScores) > 0:
            textScoresNorm = (textScores - textScores.min() + 0.1) / (textScores.max() - textScores.min() + 0.5)
        index_weights = {}
        for i in range(max(len(vectIdxs), len(textIdxs))):
            if i < len(vectIdxs):
                if vectIdxs[i] not in index_weights:
                    index_weights[vectIdxs[i]] = vectorScores[i]
                else:
                    index_weights[vectIdxs[i]] += vectorScores[i]
            if i < len(textIdxs):
                if textIdxs[i] not in index_weights:
                    index_weights[textIdxs[i]] = textScoresNorm[i]
                else:
                    index_weights[textIdxs[i]] += textScoresNorm[i]

        resIdxs = np.array(list(index_weights.keys()))
        resScores = np.array(list(index_weights.values()))
        sortedScores = np.argsort(resScores)

        return resIdxs[sortedScores[::-1]]

    def prepareDocsAndUrlsMonolitDb(self, queries, kDocuments, sqlConnection, indexDb, redundantParam = None):
        if len(queries) == 0:
            return None

        startTime = time.time()
        vectorIdxsForQueries, vectorScores = findVectorsIndexes(queries, self.modelEncoder, kDocuments, indexDb) # Поиск обратным индексом в векторной БД
        self.logDetails(f"векторный_обратный_индекс ", startTime)
        startTime = time.time()
        textIdxsForQueries, textScores, minScore, maxScore = findDocIndexesByTextSearch(queries, self.textDB, 2) # Поиск обратным индексом в текстовой БД
        combinedIndexes = self.combineVectorAndTextIndexes(vectorIdxsForQueries, vectorScores, textIdxsForQueries, textScores)
        if self.data_base_type == "monolit":
            self.logDetails(f"текстовый_обратный_индекс ", startTime)
            self.logDetails(f"scores {minScore} {maxScore}", startTime)
        if len(combinedIndexes) == 0:
            return None

        startTime = time.time()
        urlsAndDocs = get_rows_from_sql(combinedIndexes, sqlConnection, self.useStemming).fillna('stub')
        #urlsAndDocs = get_rows_from_sql(vectorIdxsForQueries, sqlConnection, self.useStemming).fillna('stub')
        if self.data_base_type == "monolit":
            self.logDetails(f"бд_с_метаданными ", startTime)
        return urlsAndDocs

    def prepareDocsAndUrlsShardedDb(self, queries, kDocuments):
        if len(queries) == 0:
            return None

        startTime = time.time()
        docsPerShard = int(kDocuments / self.shards_count)
        results = []
        with ThreadPoolExecutor() as executor:
            processedDatabases = {
                executor.submit(self.prepareDocsAndUrlsMonolitDb,
                    queries,
                    docsPerShard,
                    self.sqlConnectionSharded[dbNumber],
                    self.indexDbSharded[dbNumber]
                ) : dbNumber for dbNumber in range(self.shards_count)
            }
            for procDb in processedDatabases:
                results.append(procDb.result())

        resultsPerQueries = []
        for queryInd in range(len(queries)):
            resultsPerQueries.append(
                pd.concat([res[queryInd:queryInd+docsPerShard] for res in results])
            )
        self.logDetails(f"векторная_и_текстая_бд ", startTime)
        return pd.concat(resultsPerQueries)

    def rankDocs(self, query, docs, fullDocs, urls, ranker):
        doc_scores = ranker.rankDocuments(query, docs)
        sorted_idx = np.argsort(doc_scores)
        resDocs, resFullDocs, resUrls = list(np.array(docs)[sorted_idx[::-1]]), fullDocs.iloc[sorted_idx[::-1]], urls.iloc[sorted_idx[::-1]]
        return resDocs, resFullDocs, resUrls

    def rankAndSummarize(self, query, urlsAndDocs: pd.DataFrame, ranker = None, ranker2 = None):
        startTime = time.time()
        urls, docs, fullDocs = urlsAndDocs[0], urlsAndDocs[1], urlsAndDocs[2]
        docs = documents_filter_quorum(query, docs, stem if self.useStemming else None, self.quorum_threshold)
        self.logDetails(f"кворум {len(docs) }", startTime)
        if len(docs) == 0:
            return query, f"Ответ на вопрос {query}: ничего не найдено", list()

        resDocs, resFullDocs, resUrls = docs, fullDocs, urls
        if ranker is not None:
            startTime = time.time()
            resDocs, resFullDocs, resUrls = self.rankDocs(clean_string(query, []), docs, fullDocs, urls, ranker)
            self.logDetails(f"ранжирование ", startTime)

        if ranker2 is not None and int(len(resDocs)*self.partForRanker2) + 1 > 1:
            rank2DocCount = int(len(resDocs)*self.partForRanker2) + 1
            startTime = time.time()
            resDocs, resFullDocs, resUrls = self.rankDocs(clean_string(query, []), resDocs[:rank2DocCount], resFullDocs[:rank2DocCount], resUrls[:rank2DocCount], ranker2)
            self.logDetails(f"ранжирование_2 ", startTime)

        resFullDocs, resUrls = list(resFullDocs), list(resUrls)[:len(resDocs)]
        if self.noSummarize:
            summary = '-'
        else:
            startTime = time.time()
            summarizeCount = 5
            summary = self.summarizeText(resFullDocs[:summarizeCount], query)
            self.logDetails(f"суммаризация ", startTime)

        response_message = f"Ответ на вопрос {query}: {summary} \n\n" + '\n'.join(resUrls)
        return query, response_message, resUrls

    def sendResponse(self, responses):
        response_message = "\n\n".join(responses)
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(response_message.encode('utf-8'))

    def processQuery(self, allQueries):
        queries, responses = self.splitAllQueries(allQueries)
        # self.logDetails(f"Для обработки получено вопросов {len(queries)}")

        if self.data_base_type == "monolit":
            urlsAndDocs = self.prepareDocsAndUrlsMonolitDb(queries, self.kDocs, self.sqlConnectionMonolit, self.indexDbMonolit)
        else:
            self.sqlConnectionMonolit.close()
            urlsAndDocs = self.prepareDocsAndUrlsShardedDb(queries, self.kDocs)
        if urlsAndDocs is None:
            return [] if self.isItTest else self.sendResponse(["Ничего не нашло"])

        # Параллельное ранжирование и суммаризация для каждого вопроса
        resUrls = []
        with ThreadPoolExecutor() as executor:
            processed_queries = {
                executor.submit(self.rankAndSummarize, query, urlsAndDocs[idxStart:idxStart+self.kDocs], self.ranker, self.ranker2): (query, idxStart)
                for query, idxStart in zip(queries, range(0, self.kDocs*len(queries), self.kDocs))
            }
            for proc_query in processed_queries:
                response = proc_query.result()[1]
                urls = proc_query.result()[2]
                responses.append(response)
                resUrls.extend(urls)
                # self.query_history[question] = "Сохранённый ответ.\n" + response

        # self.logDetails(f"Вопросов {len(queries)} обработано за ", startTime)
        if self.isItTest:
            return resUrls
        else:
            self.sendResponse(responses)

    def do_GET(self):
        # Получаем вопрос из url
        path = self.path
        parts = path.split('/')
        # Если запрос начинается с нужного префикса, обрабатываем его
        if len(parts) < 2:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            return
        query = unquote(parts[1].replace("_", " "))

        self.processQuery(query)

    def do_POST(self):
        # Получаем вопрос из тела
        content_length = int(self.headers['Content-Length'])
        query = self.rfile.read(content_length).decode("utf-8")

        self.processQuery(query)

    data_base_type = "monolit"
    # data_base_type = "sharded"
    shards_count = 9

    vectorDBPathMonolit = "text_parser/data/data_bases/monolit/vectorDB.index"
    vectorDBPathSharded = [f"text_parser/data/data_bases/sharded/vectorDB_{shard}.index" for shard in range(shards_count)]
    textDB = open_dir("text_parser/data/data_bases/text_index")
    metadataDBPathMonolit = "text_parser/data/data_bases/monolit/documentsMetadataDB.db"
    metadataDBPathSharded = [f"text_parser/data/data_bases/sharded/documentsMetadataDB_{shard}.db" for shard in range(shards_count)]
    modelEncoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    tokenizer = T5TokenizerFast.from_pretrained('utrobinmv/t5_summary_en_ru_zh_base_2048')
    modelSummarizer = T5ForConditionalGeneration.from_pretrained('utrobinmv/t5_summary_en_ru_zh_base_2048')
    # tokenizer = T5TokenizerFast.from_pretrained('UrukHan/t5-russian-summarization')
    # modelSummarizer = AutoModelForSeq2SeqLM.from_pretrained('UrukHan/t5-russian-summarization')

    sqlConnectionMonolit = sqlite3.connect(metadataDBPathMonolit, check_same_thread=False)
    # sqlConnectionSharded = [sqlite3.connect(path, check_same_thread=False) for path in metadataDBPathSharded]
    indexDbMonolit = faiss.read_index(vectorDBPathMonolit)
    # indexDbSharded = [faiss.read_index(path) for path in vectorDBPathSharded]

    queryHistory = dict()
    useStemming = True
    isItTest = False
    ranker = None
    # ranker = Bm25Ranker(bm25_alg = BM25WithProximity, preprocess_func = stem if useStemming else None)
    # ranker = CrossEncoderRanker()
    # ranker = BiEncoderRanker()
    # ranker2 = CrossEncoderRanker()
    # ranker2 = BiEncoderRanker()
    ranker2 = None
    partForRanker2 = 0.1
    quorum_threshold = 0.0
    kDocs = 400
    noSummarize = False

def run(server_class=HTTPServer, handler_class=Poiskovik, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Запуск сервера на порте {port}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
