# 环境没配好，这个没跑通
from allennlp.predictors.predictor import Predictor

# 加载预训练的实体识别模型
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")

# 示例文本
text = """
The system should allow users to securely access their personal information. 
Users should be able to authenticate using their email and password. 
The system should store sensitive data, such as credit card information, using encryption.
"""

# 使用AllenNLP模型进行实体识别
result = predictor.predict(sentence=text)

# 输出识别到的实体及其标签
for word, tag in zip(result["words"], result["tags"]):
    if tag != "O":
        print(word, tag)