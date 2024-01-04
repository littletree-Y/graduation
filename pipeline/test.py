from util import *
from const import *
from tfidf import *
from doc2vec import *
import random

if __name__ == "__main__":
    # etcs_dicts = read_example_dict(etcs_file)
    # cwe_dicts = read_example_dict(cwe_file)
    # etcs = [tmp["text"] for tmp in etcs_dicts]
    # cwes = [tmp["Name"] for tmp in cwe_dicts]
    # # matched_indices, scores = get_top_k(etcs, cwes, 2)
    # matched_indices, scores = get_top_k_doc2vec(etcs, cwes, 2)
    # # print(matched_indices[:1])
    # # print(scores[0:1])
    # random_list = random.sample(range(len(etcs)), 5)
    # for i in random_list:
    #     print(etcs[i])
    #     for j in matched_indices[i]:
    #         print(cwes[j])
    #     print("="*10)

    import nltk
    from rake_nltk import Rake

    # 首先下载一次nltk的停用词列表，只需下载一次
    # nltk.download('stopwords')

    # 要分析的句子
    # text = "If the PSAM does not have an aggregation record for the card issuer, a new aggregation record for the card issuer must be created"
    # text = "The system shall be able to hash passwords"
    text = "The system shall allow generation of public and private keys.  These keys will be used for encryption and decryption of the AES symmetric key for a safer key exchange"
   # 自定义停用词
    my_stopwords = set(nltk.corpus.stopwords.words('english'))
    my_stopwords.update(['system', 'shall'])

    # 使用Rake算法，同时传入自定义的停用词
    rake_nltk_var = Rake(stopwords=my_stopwords)

    # 提取关键词
    rake_nltk_var.extract_keywords_from_text(text)

    # 获取关键词短语排名
    keyword_ranked = rake_nltk_var.get_ranked_phrases()

    # 打印出排名靠前的关键词短语
    print(keyword_ranked)