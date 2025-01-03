import faiss
import numpy as np
import pandas as pd
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import QueryParser
from whoosh import scoring
from nltk.stem.snowball import SnowballStemmer
import re
import gensim
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
nltk.download('stopwords')
from concurrent.futures import ThreadPoolExecutor

sw_ru = set(stopwords.words('russian'))

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    # reg = re.compile('[^\wёЁ]')
    return reg.sub(' ', string)


def _remove_stop_words(string, sw=[]):
    return ' '.join([word if word not in sw else '' \
                     for word in string.strip().split(' ')])


def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case
    return re.sub('\s+', ' ', string).strip().lower()


def clean_string(string,
                 stop_words_list,
                 min_len=2,
                 max_len=30):
    string = _remove_non_printed_chars(string)
    string = _remove_stop_words(string, stop_words_list)
    string = _trim_string(string)
    # also remove short words, most likely containing addresses / crap / left-overs / etc remaining after removal
    # gensim mostly does the same as above, it is used here for simplicity
    string = ' '.join(gensim.utils.simple_preprocess(string,
                                                     min_len=min_len,
                                                     max_len=max_len))
    return string

def stem(doc):
   stemmer = SnowballStemmer("russian")
   return [stemmer.stem(word) for word in doc.split()]

def get_rows_from_sql(indexes, connection, useStemming):
    cursor = connection.cursor()
    subqueries = []
    processed_articles_column = 'stem_article' if useStemming else 'proc_article'

    # Вариант для обработки нескольких запросов обновременно
    for index in indexes:
        subqueries.append(f"SELECT url, {processed_articles_column}, article FROM documents WHERE \"index\" = {index}")
    queries = [' UNION ALL '.join(subqueries[i:i + 400]) for i in range(0, len(subqueries), 400)]

    res = list()
    for query in queries:
        cursor.execute(query)
        res.extend(cursor.fetchall())
    return pd.DataFrame(res)

    # indexes_str = ', '.join(str(ind) for ind in indexes)
    # query = f"SELECT url, {processed_articles_column}, article FROM documents WHERE \"index\" IN ({indexes_str})"
    # cursor.execute(query)
    # return pd.DataFrame(cursor.fetchall())

def getVectorDB(path):
    return faiss.read_index(path)

def findVectorsIndexes(queries, encoder, kDocuments, index):
    queryEmbd = encoder.encode(queries, normalize_embeddings=True)
    D, I = index.search(np.array(queryEmbd), kDocuments)
    return I[:len(queries)].flatten(), 1/(1+4*(D[:len(queries)]**2).flatten())

def textSearch(searcher, parsedCondition, lvl):
    return [searcher.search(parsedCondition)], lvl

def findDocIndexesByTextSearch(queries : list[str], textIndex, depthLevels = 0):
    global sw_ru

    parser = QueryParser("content", textIndex.schema)
    searcher = textIndex.searcher(weighting=scoring.BM25F())
    indexesAndScores = []
    searchCndsByLevels = []
    for query in queries:
        words = stem(clean_string(query, sw_ru))
        lenQuery = len(words)
        searchCndsByLevels.append((0, " AND ".join(words)))
        # for lvl in range(lenQuery-depthLevels, lenQuery):
        for lvl in range(1, depthLevels+1):
            operands = lenQuery - lvl
            if lvl < 1: break
            searchCndsByLevels.extend(
                [(lvl, " AND ".join(words[i:i+operands])) for i in range(lenQuery-operands+1)]
            )

    with ThreadPoolExecutor() as executor:
        processed_cnd = {
            executor.submit(textSearch, searcher, parser.parse(conditionStr), lvl): (lvl, conditionStr)
            for lvl, conditionStr in searchCndsByLevels
        }
        for proc_cnd in processed_cnd:
            lvl = proc_cnd.result()[1]
            indexesAndScores.extend([
                [float(result['title']), 1/(1+lvl)*float(result.score)] for result in proc_cnd.result()[0][0]
            ])

    if len(indexesAndScores) == 0:
        return np.array([]), np.array([]), 0, 0
    indexesAndScores = np.array(indexesAndScores)
    sorted_idx = np.argsort(indexesAndScores[:, 1])[::-1]
    return indexesAndScores[sorted_idx, 0], indexesAndScores[sorted_idx, 1], np.min(indexesAndScores[:, 1]), np.max(indexesAndScores[:, 1])  #indexesAndScores[:, 0][sorted_idx[::-1]]

def get_rows_from_csv(filename, indices):
    df = pd.read_csv(
        filename,
        header=None,
        skiprows=lambda x: x not in indices
    )
    return df