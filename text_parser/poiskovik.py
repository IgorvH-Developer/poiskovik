from http.server import BaseHTTPRequestHandler, HTTPServer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from abc import ABC, abstractmethod
from urllib.parse import unquote


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
    def get_rows_from_csv(self, filename, indices):
        df = pd.read_csv(
            filename,
            header=None,
            skiprows=lambda x: x not in indices
        )
        return df

    def getVectorDB(self, path):
        return faiss.read_index(path)

    def findVectorsIndexes(self, query, encoder, kDocuments):
        index = self.getVectorDB(self.vectorDBPath)
        queryEmbd = encoder.encode(query, normalize_embeddings=True)
        D, I = index.search(np.array([queryEmbd]), kDocuments)
        return I[0]

    def retrieveDocsAndUrls(self, indexes):
        urlsAndDocs = self.get_rows_from_csv(self.textsCsvPath, indexes)[[2, 5]]
        urlsAndDocs = urlsAndDocs.fillna('stub')
        return urlsAndDocs[2], urlsAndDocs[5]

    def rankDocuments(self, query, indexes, ranker):
        urls, docs = self.retrieveDocsAndUrls(indexes)
        doc_scores = ranker.rankDocuments(query, docs)
        sorted_idx = np.argsort(doc_scores)
        return list(docs.iloc[sorted_idx[::-1]]), list(urls.iloc[sorted_idx[::-1]]), doc_scores[sorted_idx[::-1]]

    def getSortedDocumentsWithUrls(self, query, encoder, kDocuments, ranker):
        indexes = self.findVectorsIndexes(query, encoder, kDocuments)
        return self.rankDocuments(query, indexes, ranker)

    def sendAnswer(self, query):
        kDocuments = 50
        # docs, urls, bm25_scores = self.getSortedDocumentsWithUrls(query, self.modelEncoder, kDocuments, Bm25Ranker())
        # docs, urls, bm25_scores = self.getSortedDocumentsWithUrls(query, self.modelEncoder, kDocuments, Bm25Ranker(lemmatize))
        # docs, urls, bm25_scores = self.getSortedDocumentsWithUrls(query, self.modelEncoder, kDocuments, Bm25Ranker(stem))
        docs, urls, scores = self.getSortedDocumentsWithUrls(query, self.modelEncoder, kDocuments, CrossEncoderRanker())

        response_message = f"Лучший ответ: {docs[0]} \n\n"
        response_message += '\n'.join(urls)

        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()

        self.wfile.write(response_message.encode('utf-8'))

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

    vectorDBPath = "./data/data_bases/vectorDB.index"
    textsCsvPath = "./data/data_bases/texts.csv"
    modelEncoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

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


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()