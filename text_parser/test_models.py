import os
import re
import sys
import glob
import gensim
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from uuid import uuid4
from functools import reduce
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.corpus import stopwords
import sqlite3

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from sklearn.feature_extraction.text import TfidfVectorizer

import wikiextractor


def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub(' ', string)

def _remove_stop_words(string,sw=[]):
    return ' '.join([word if word not in sw else '' \
                     for word in string.strip().split(' ')])

def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case 
    return re.sub('\s+',' ',string).strip().lower()
    
def clean_string(string,
                 stop_words_list,
                 min_len=2,
                 max_len=30):

    string = _remove_non_printed_chars(string)
    string = _remove_stop_words(string,stop_words_list)
    string = _trim_string(string)
    # also remove short words, most likely containing addresses / crap / left-overs / etc remaining after removal
    # gensim mostly does the same as above, it is used here for simplicity
    string = ' '.join(gensim.utils.simple_preprocess(string,
                                                     min_len=min_len,
                                                     max_len=max_len))
    return string
    
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    
def remove_special_chars(text,char_list):
    for char in char_list:
        text=text.replace(char,'')
    return text.replace(u'\xa0', u' ')

def splitkeepsep(s, sep):
    cleaned = []
    s = re.split("(%s)" % re.escape(sep), s)
    for _ in s:
        if _!='' and _!=sep:
            cleaned.append(sep+_)
    return cleaned

def extract_url(text):
    pattern = 'http([^"]+)'
    match = re.search(pattern, text)
    if match:
        url = match.group(0)
        return url
    else:
        return ""

def create_vector(text):
    return model.encode(text, normalize_embeddings=True)


import requests
from bs4 import BeautifulSoup

# URL статьи Википедии
url = 'https://ru.wikipedia.org/wiki?curid=9'

def getHeadings(url):
    # Получаем содержимое страницы
    response = requests.get(url)
    
    # Парсим HTML-код с помощью BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Находим элемент с оглавлением (обычно он находится внутри элемента с классом mw-parser-output)
    toc = soup.find('div', id='toc')
    if toc is None:
        return None
    
    # Извлекаем все элементы списка (<li>) из оглавления
    items = toc.find_all('li')
    
    # Формируем список заголовков
    headings = []
    for item in items:
        link = item.find('a')  # находим ссылку внутри каждого пункта списка
        if link is not None:
            heading_text = link.text.strip()  # получаем текст ссылки
            cleaned_heading = heading_text.split(maxsplit=1)[-1].strip()  # убираем номер и точку
            if cleaned_heading + '.' not in headings:
                headings.append(cleaned_heading + '.')  # добавляем очищенное название в список
    
    # Выводим результат
    return headings


def process_wiki_files(wiki_file):
    chars = ['\n\n']
    global sw

    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()

    articles = splitkeepsep(content,'<doc id=')
    df_texts = pd.DataFrame(columns=['article_uuid','url', 'title', 'article','proc_article','proc_len'])
    emds = []

    for article in articles:
        if len(article) < 500:
            continue

        uuid_text = uuid4()
        
        articleParts = article.split('\n')
        url = extract_url(article)
        headings = getHeadings(url)
        if headings is None:
            continue
        title = articleParts[1]

        article = remove_html_tags(article)
        article = remove_special_chars(article, chars)
        clearArticleParts = article.split('\n')
        
        startIndex = 1
        currHeading = ''
        
        for endIndex in range(startIndex, len(clearArticleParts)):
            if len(clearArticleParts[endIndex]) < 100 and clearArticleParts[endIndex] in headings: 
                if endIndex - startIndex == 1:
                    startIndex = endIndex
                    currHeading = clearArticleParts[endIndex]
                    continue
            
                onePart = title + '. ' + currHeading + ' ' + ' '.join(clearArticleParts[startIndex+1:endIndex])
            
                proc_onePart = clean_string(onePart, sw_ru)
                proc_len = len(proc_onePart.split(' '))
            
                temp_df_texts = pd.DataFrame(
                    {'article_uuid': [uuid_text],
                     'url': url + "#" + currHeading[:-1].replace(' ', '_') if len(currHeading) > 0 else url,
                     'title': title + '. ' + currHeading if len(currHeading) > 0 else title,
                     'article': onePart,
                     'proc_article':proc_onePart,
                     'proc_len':proc_len
                    })
                df_texts = pd.concat([df_texts, temp_df_texts], ignore_index=True)
            
                emb = create_vector(proc_onePart)
                emds.append(emb)
            
                startIndex = endIndex
                currHeading = clearArticleParts[endIndex]
    
    return df_texts, np.array(emds)

sw_en = set(stopwords.words('english'))
sw_ru = set(stopwords.words('russian'))
sw = list(sw_ru.union(sw_en))


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


import os
import faiss
from os.path import exists

def saveEmbdsToVectorDB(embds, path):
    if not exists(path):
        index = faiss.IndexFlatL2(embds.shape[1]) 
        index = faiss.IndexIDMap(index)
        index.add_with_ids(embds, np.arange(0, embds.shape[0]))
        faiss.write_index(index, path)
    else:
        index = faiss.read_index(path)
        index.add_with_ids(embds, np.arange(index.ntotal, index.ntotal + embds.shape[0]))
        faiss.write_index(index, path)


def getVectorDB(path):
    return faiss.read_index(path)

def addMetadataToDB(pathDB, cursor, conn, metadataDf):
    metadataDf.to_sql(name='documents', con=conn, if_exists='append', index=False)
    conn.commit()


def getRevertedIndexTextDB(pathDB):
    return open_dir(pathDB)


def get_rows_from_csv(filename, indices):
    df = pd.read_csv(
        filename,
        header=None,
        skiprows=lambda x: x not in indices
    )
    
    return df


def textSearch_with_bm25_ranking(query, pathDB):
    index = getRevertedIndexTextDB(pathDB)
    with index.searcher() as searcher:
        query_parser = QueryParser("content", index.schema)
        parsed_query = query_parser.parse(query)
        print("Получился запрос вида: ", parsed_query)
        results = searcher.search(parsed_query)
        return np.array([(result['id'], result.score) for result in results])


wikiFilesRootPath = "data/wiki"
vectorDBPath = 'data/data_bases/monolit/vectorDB.index'
metadataDBPath = "data/data_bases/monolit/documentsMetadataDB.db"
textsCsvPath = "data/data_bases/monolit/texts.csv"

import nltk
from nltk.stem.snowball import SnowballStemmer
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


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

from rank_bm25 import BM25Okapi
import pymorphy2

def findVectorsIndexes(query, encoder, kDocuments):
  index = getVectorDB(vectorDBPath)
  queryEmbd = encoder.encode(query, normalize_embeddings=True)
  D, I = index.search(np.array([queryEmbd]), kDocuments)
  return I[0]

def findVectorsDistsAndIndexes(query, encoder, kDocuments):
  index = getVectorDB(vectorDBPath)
  queryEmbd = encoder.encode(query, normalize_embeddings=True)
  D, I = index.search(np.array([queryEmbd]), kDocuments)
  return D[0], I[0]

sqlConnection = sqlite3.connect(metadataDBPath)

def get_rows_from_sql(indices):
    def get_row_by_position(curs, table_name, position):
        curs.execute(f'SELECT url, proc_article FROM {table_name} LIMIT 1 OFFSET ?', (position - 1,))
        return curs.fetchone()
    cursor = sqlConnection.cursor()
    rows = [get_row_by_position(cursor, 'documents', int(idx)) for idx in indices]
    return pd.DataFrame(rows)

def retrieveDocsAndUrls(indexes):
    #urlsAndDocs = get_rows_from_csv(textsCsvPath, indexes)[[2, 5]]
    urlsAndDocs = get_rows_from_sql(indexes)
    urlsAndDocs = urlsAndDocs.fillna('stub')
    #return urlsAndDocs[2], urlsAndDocs[5]
    return urlsAndDocs[0], urlsAndDocs[1]

def rankDocuments(query, indexes, ranker):
    urls, docs = retrieveDocsAndUrls(indexes)
    doc_scores = ranker.rankDocuments(query, docs)
    sorted_idx = np.argsort(doc_scores)
    return list(docs.iloc[sorted_idx[::-1]]), list(urls.iloc[sorted_idx[::-1]]), doc_scores[sorted_idx[::-1]]

def getSortedDocumentsWithUrls(query, encoder, kDocuments, ranker):
  indexes = findVectorsIndexes(query, encoder, kDocuments)
  return rankDocuments(query, indexes, ranker)

def getUnsortedDocumentsWithUrls(query, encoder, kDocuments):
  indexes = findVectorsIndexes(query, encoder, kDocuments)
  return retrieveDocsAndUrls(indexes)


def test_model(filename, ranker = None, document_num = 50):
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
    if ranker is None:
        N = len(queries)
        for i in range(N):
            start_time = time.time()
            indexes = findVectorsIndexes(queries[i], model, document_num)
            end_time = time.time()
            time_arr[i] = end_time - start_time
            urls, docs = retrieveDocsAndUrls(indexes)
            for j in range(urls.shape[0]):
                if urls[j] == real_urls[i]:
                    pos_arr[i] = j
                    break
    else:
        N = len(queries)
        for i in range(N):
            start_time = time.time()
            dists, indexes = findVectorsDistsAndIndexes(queries[i], model, document_num)
            end_time = time.time()
            time_arr[i] = end_time - start_time
            urls, docs = retrieveDocsAndUrls(indexes)
            start_time = time.time()
            doc_scores = ranker.rankDocuments(queries[i], docs)
            sorted_idx = np.argsort(doc_scores)
            end_time = time.time()
            anses = urls.iloc[sorted_idx[::-1]].to_numpy()
            time_arr[i] += end_time - start_time
            for j in range(anses.shape[0]):
                if anses[j] == real_urls[i]:
                    pos_arr[i] = j
                    break
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


def eval_model(filename, ranker = None, echo = False, document_num = 50, metric = metric_inv):
    p, t = test_model(filename, ranker = ranker, document_num = document_num)
    if echo:
        print(p)
        print(t)
    for i in range(p.shape[0]):
        p[i] = metric(p[i])
    return {'score' : p.mean(), 'avg_t' : t.mean(), 'std_t' : t.std()}


def multi_eval(filenames, rankers, document_nums, metrics):
    for filename in filenames:
        for ranker in rankers:
            for document_num in document_nums:
                for metric in metrics:
                    print(f'filename: {filename[0]}, ranker: {ranker[0]}, doc num: {document_num}, metric: {metric[0]}, ')
                    print(eval_model(filename[1], ranker = ranker[1], document_num = document_num, metric = metric[1]))


def general_test(filenames, rankers, document_nums, metrics):
    for ranker in rankers:
        for document_num in document_nums:
            scores = np.zeros((len(filenames), len(metrics)))
            times = np.zeros(len(filenames))
            for k in range(len(filenames)):
                filename = filenames[k]
                p, t = test_model(filename[1], ranker = ranker[1], document_num = document_num)
                for i in range(len(metrics)):
                    scored = np.zeros(p.shape[0])
                    for j in range(p.shape[0]):
                        scored[j] = metrics[i][1](p[j])
                    scores[k][i] = scored.mean()
                times[k] = t.mean()
            print(f'ranker: {ranker[0]}, doc num: {document_num}')
            times = times.mean()
            scores = scores.mean(axis = 0)
            res_str = f'time: {times}'
            for i in range(len(metrics)):
                res_str += f' {metrics[i][0]}: {scores[i]}'
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
                filenames.append([word, 'data/queries_split/' + word + '.txt'])
            for word in lines[1][:-1].split(' '):
                if word == 'None':
                    rankers.append([word, None])
                if word == 'BM25':
                    rankers.append([word, Bm25Ranker()])
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
