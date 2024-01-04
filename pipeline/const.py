"""
记录经常用到的一些变量定义
"""
import nltk

cwe_file = "../data/cwe/cwe.jsonl"
etcs_file = "../data/requirement/etcs.jsonl"

custom_stopwords = ["system", "shall"]
my_stopwords = set(nltk.corpus.stopwords.words('english'))
my_stopwords.update(custom_stopwords)