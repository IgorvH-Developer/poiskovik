from http.server import BaseHTTPRequestHandler, HTTPServer
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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

    def getVectorsIndexes(self, query, encoder, kDocuments):
        index = self.getVectorDB(self.vectorDBPath)
        queryEmbd = encoder.encode(query, normalize_embeddings=True)
        D, I = index.search(np.array([queryEmbd]), kDocuments)
        return I[0]

    def rankDocumentsbyBm25(self, query, indexes):
        urlsAndDocs = self.get_rows_from_csv(self.textsCsvPath, indexes)[[2, 5]]
        urlsAndDocs = urlsAndDocs.fillna('stub')
        tokenized_corpus = [doc.split() for doc in urlsAndDocs[5]]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        doc_scores = bm25.get_scores(tokenized_query)
        sorted_idx = np.argsort(doc_scores)
        # sorted_idx = list(range(len(indexes)))[::-1]  # это чтобы посмотреть на результат без сортировки BM25
        return list(urlsAndDocs[5].iloc[sorted_idx[::-1]]), list(urlsAndDocs[2].iloc[sorted_idx[::-1]]), doc_scores[sorted_idx[::-1]]

    def getSortedDocumentsWithUrls(self, query, encoder, kDocuments):
        indexes = self.getVectorsIndexes(query, encoder, kDocuments)
        return self.rankDocumentsbyBm25(query, indexes)

    def do_GET(self):
        # Отправляем стандартный ответ
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        message = "<html><body><h1>Hello, World!</h1></body></html>"
        self.wfile.write(bytes(message, "utf8"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode("utf-8")

        docs, urls,  _ = self.getSortedDocumentsWithUrls(post_data, self.modelEncoder, 50)

        print(f"Принято сообщение: {post_data}")

        response_message = f"Лучший ответ: {docs[0]} \n\n"
        response_message += '\n'.join(urls)

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        self.wfile.write(response_message.encode())

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