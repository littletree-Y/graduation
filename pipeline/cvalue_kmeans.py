"""
想实现这样一个功能，有很多缺陷条目，
首先用c value的方式提取出其中的名词等，
再用聚类的方式将这些名词归类，并得到每个类别的表示
"""
import spacy
import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np

# 加载预训练的spaCy英文模型
#python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

# 示例缺陷条目
defect_entries = [
    "The system fails to validate user input, leading to SQL injection vulnerability.",
    "Memory leak occurs when the application does not release memory after use.",
    "The authentication process is bypassed due to improper session management.",
    # 添加更多缺陷条目
]

# 使用C-Value方法提取名词短语
noun_phrases = []
for entry in defect_entries:
    doc = nlp(entry)
    noun_phrases.extend([chunk.text for chunk in doc.noun_chunks])

# 加载预训练的GloVe词嵌入模型
model = api.load("glove-wiki-gigaword-50")

# 将名词短语转换为向量表示
noun_phrase_vectors = []
for phrase in noun_phrases:
    words = phrase.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        phrase_vector = np.mean(word_vectors, axis=0)
        noun_phrase_vectors.append(phrase_vector)

# 使用K-Means聚类算法对名词短语向量进行聚类
kmeans = KMeans(n_clusters=3)  # 设置聚类数量，可以根据需要调整
kmeans.fit(noun_phrase_vectors)

# 分析每个聚类中的名词短语，以确定缺陷类别
for cluster_id in range(kmeans.n_clusters):
    cluster_phrases = [noun_phrases[i] for i, label in enumerate(kmeans.labels_) if label == cluster_id]
    print(f"Cluster {cluster_id}: {', '.join(cluster_phrases)}")