

def get_tfidf_matrix(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 使用TfidfVectorizer类将文本集合转换为TF-IDF特征矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    return tfidf_matrix

def cos_sim(doc1, doc2):
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(doc1, doc2)
    return similarity


def get_top_k(a, b, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from nltk.corpus import stopwords
    """
    输入两个文档列表，对a中的每个文档，选择最匹配的top_k个文档
    """
    # 将a和b合并成一个列表
    docs = a + b

    # 去除停用词
    stop_words = set(stopwords.words('english'))
    docs = [' '.join([word for word in doc.split() if word.lower() not in stop_words]) for doc in docs]


    # 使用TfidfVectorizer类将文本集合转换为TF-IDF特征矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    # 计算文本余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # 为a中的每个文本匹配top-k个b中的文本
    matched_indices = np.argsort(-similarity_matrix[:len(a), len(a):], axis=1)[:, :k]
    similarity_scores = np.sort(-similarity_matrix[:len(a), len(a):], axis=1)[:, :k]

    similarity_scores = -similarity_scores
    return matched_indices, similarity_scores