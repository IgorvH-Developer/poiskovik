from http.server import BaseHTTPRequestHandler, HTTPServer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from abc import ABC, abstractmethod
from urllib.parse import unquote
import sqlite3
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import nltk
from nltk.stem.snowball import SnowballStemmer
import pymorphy2
import time
import logging
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования
logging.basicConfig(
    filename='logs/queries.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocsRanker(ABC):
    @abstractmethod
    def rankDocuments(self, query, docs):
        pass

class Bm25Ranker(DocsRanker):
    # preprocess_func: переобразует запрос и документ в список слов
    def __init__(self, preprocess_func = None) -> None:
        self.preprocess_func = preprocess_func

    def rankDocuments(self, query, docs):
        if self.preprocess_func is None:
            self.preprocess_func = lambda doc: doc.split()
        tokenized_corpus = [self.preprocess_func(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.preprocess_func(query)
        return bm25.get_scores(tokenized_query)

def lemmatize(doc):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(word)[0].normal_form for word in doc.split()]

def stem(doc):
   stemmer = SnowballStemmer("russian")
   words = nltk.word_tokenize(doc, language="russian")
   return [stemmer.stem(word) for word in words]


class CrossEncoderRanker(DocsRanker):
    def __init__(self) -> None:
        # self.reranker_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device='cuda')
        self.reranker_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device='cpu')

    def rankDocuments(self, query, docs):
        return np.array([self.reranker_model.predict([[query, doc]])[0] for doc in docs])

class Poiskovik(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def log_details(self, text):
        logging.info(f"{text}")

    def log_details(self, text, start_time):
        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"{text} {processing_time:.2f} секунд.")

    def summarizeText(self, docs, query=None):
        max_input = 1512
        task_prefix = "Find answer on question: "
        input_seq = ""
        if query != None:
            input_seq = " " + query + "\n" + "В тексте: "
        input_seq += "\n".join(docs)
        input_seq = [input_seq]
        encoded = self.tokenizer(
            [task_prefix + sequence for sequence in input_seq],
            padding="longest",
            max_length=max_input,
            truncation=True,
            return_tensors="pt",
        )
        predicts = self.modelSummarizer.generate(encoded['input_ids'])  # # Прогнозирование
        return self.tokenizer.batch_decode(predicts, skip_special_tokens=True)  # Декодируем данные

    def get_rows_from_csv(self, filename, indices):
        df = pd.read_csv(
            filename,
            header=None,
            skiprows=lambda x: x not in indices
        )
        return df

    def get_rows_from_sql(self, indexes):
        cursor = self.sqlConnection.cursor()
        subqueries = []
        for index in indexes:
            subqueries.append(f"SELECT url, proc_article, article FROM documents WHERE \"index\" = {index}")
        query = ' UNION ALL '.join(subqueries)

        # conditions = ' OR '.join(f'"index" = {ind}' for ind in indexes)
        # query = f'SELECT url, proc_article, article FROM documents WHERE {conditions}'

        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

    def getVectorDB(self, path):
        return faiss.read_index(path)

    def findVectorsIndexes(self, query, encoder, kDocuments):
        index = self.getVectorDB(self.vectorDBPath)
        queryEmbd = encoder.encode(query, normalize_embeddings=True)
        D, I = index.search(np.array(queryEmbd), kDocuments)
        return I[:len(query)]

    def retrieveDocsAndUrls(self, indexes):
        # urlsAndDocs = self.get_rows_from_csv(self.textsCsvPath, indexes)[[2, 4, 5]]
        urlsAndDocs = self.get_rows_from_sql(indexes)
        urlsAndDocs = urlsAndDocs.fillna('stub')
        # return urlsAndDocs[2], urlsAndDocs[5], urlsAndDocs[4]
        return urlsAndDocs[0], urlsAndDocs[1], urlsAndDocs[2]

    def rankDocuments(self, query, indexes, ranker):
        urls, docs, fullDocs = self.retrieveDocsAndUrls(indexes)
        doc_scores = ranker.rankDocuments(query, docs)
        sorted_idx = np.argsort(doc_scores)
        return list(fullDocs.iloc[sorted_idx[::-1]]), list(urls.iloc[sorted_idx[::-1]]), doc_scores[sorted_idx[::-1]]

    # def getSortedDocumentsWithUrls(self, query, encoder, kDocuments, ranker):
    #     indexes = self.findVectorsIndexes(query, encoder, kDocuments)
    #     print("Векторный поиск. ГОТОВО!")
    #     return self.rankDocuments(query, indexes, ranker)

    def splitAllQueries(self, allQueries):
        queries = []
        responses = []
        for question in allQueries.split('\n\n'):
            if question in self.query_history.keys():
                responses.append(self.query_history[question])
                continue
            queries.append(question)
        return queries, responses

    def sendAnswer(self, allQueries):
        kDocuments = 50
        start_time = time.time()

        queries, responses = self.splitAllQueries(allQueries)
        logging.info(f"Для обработки получено вопросов {len(queries)}")
        indexesForQueries = self.findVectorsIndexes(queries, self.modelEncoder, kDocuments)
        self.log_details(f"Готов поиск в векторной БД ", start_time)

        urlsAndDocs = self.get_rows_from_sql(indexesForQueries.flatten()).fillna('stub')
        self.log_details(f"Готов поиск в текстовой БД ", start_time)

        with ThreadPoolExecutor() as executor:
            processed_queries = {
                executor.submit(self.process_query, query, urlsAndDocs[idxStart:idxStart+kDocuments], self.ranker, start_time): (query, idxStart)
                for query, idxStart in zip(queries, range(0, kDocuments*len(queries), kDocuments))
            }
            for proc_query in processed_queries:
                try:
                    question = proc_query.result()[0]
                    response = proc_query.result()[1]
                    responses.append(response)
                    # self.query_history[question] = "Сохранённый ответ.\n" + response
                except Exception as exc:
                    logging.error(f"Ошибка обработки запроса {proc_query}: {exc}")

        # for query, idxStart in zip(queries, range(0, kDocuments*len(queries), kDocuments)):
        #     res = self.process_query(query, urlsAndDocs[idxStart:idxStart + kDocuments], self.ranker, start_time)
        #     question = res[0]
        #     response = res[1]
        #     responses.append(response)
        #     self.query_history[question] = "Сохранённый ответ.\n" + response

        self.log_details(f"Вопросов {len(queries)} обработано за ", start_time)
        response_message = "\n\n".join(responses)

        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(response_message.encode('utf-8'))

    def process_query(self, query, urlsAndDocs: pd.DataFrame, ranker, start_time):
        urls, docs, fullDocs = urlsAndDocs[0], urlsAndDocs[1], urlsAndDocs[2]
        doc_scores = ranker.rankDocuments(query, docs)
        sorted_idx = np.argsort(doc_scores)
        resDocs, urls, scores = list(fullDocs.iloc[sorted_idx[::-1]]), list(urls.iloc[sorted_idx[::-1]]), doc_scores[sorted_idx[::-1]]
        self.log_details(f"Ранжирование готово ", start_time)

        summarizeCount = 5
        summary = self.summarizeText(resDocs[:summarizeCount], query)
        self.log_details(f"Суммаризация готова ", start_time)

        response_message = f"Ответ на вопрос {query}: {summary} \n\n" + '\n'.join(urls)
        return [query, response_message]

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
        print(f"Принято сообщение: {query}")

        self.sendAnswer(query)

    def do_POST(self):
        # Получаем вопрос из тела
        content_length = int(self.headers['Content-Length'])
        query = self.rfile.read(content_length).decode("utf-8")
        print(f"Принято сообщение: {query}")

        self.sendAnswer(query)

    vectorDBPath = "text_parser/data/data_bases/monolit/vectorDB.index"
    textsCsvPath = "text_parser/data/data_bases/monolit/texts.csv"
    metadataDBPath = "text_parser/data/data_bases/monolit/documentsMetadataDB.db"
    modelEncoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    tokenizer = T5TokenizerFast.from_pretrained('UrukHan/t5-russian-summarization')
    modelSummarizer = AutoModelForSeq2SeqLM.from_pretrained('UrukHan/t5-russian-summarization')

    sqlConnection = sqlite3.connect(metadataDBPath, check_same_thread=False)
    query_history = dict()
    # ranker = Bm25Ranker()
    # ranker = Bm25Ranker(lemmatize)
    # ranker = Bm25Ranker(stem)
    ranker = CrossEncoderRanker()

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
        print('Сервер остановлен.')
        logging.info('Сервер остановлен.')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
