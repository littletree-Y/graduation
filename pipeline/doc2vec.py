from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords

def get_top_k_doc2vec(a, b, k):
    """
    输入两个文档列表，对a中的每个文档，选择最匹配的top_k个文档
    """
    # 将a和b合并成一个列表
    docs = a + b

    # 去除停用词
    stop_words = set(stopwords.words('english'))
    docs = [' '.join([word for word in doc.split() if word.lower() not in stop_words]) for doc in docs]

    # 使用Doc2Vec模型将文本转换为向量表示
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
    doc_vectors = [model.infer_vector(doc.split()) for doc in docs]

    # 计算文本余弦相似度
    similarity_matrix = cosine_similarity(doc_vectors)

    # 为a中的每个文本匹配top-k个b中的文本
    matched_indices = np.argsort(-similarity_matrix[:len(a), len(a):], axis=1)[:, :k]
    similarity_scores = np.sort(-similarity_matrix[:len(a), len(a):], axis=1)[:, :k]

    similarity_scores = -similarity_scores
    return matched_indices, similarity_scores