# export HUGGINGFACE_CO_TOKEN=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

"""
通用的模型还是无法识别安全实体 比如私人信息，密码
"""
from transformers import pipeline

# 加载预训练的DistilBERT NER模型
nlp = pipeline("ner", model="dslim/bert-base-NER")

# 示例功能需求文本 无输出
# text = """
# The system should allow users to securely access their personal information. 
# Users should be able to authenticate using their email and password. 
# The system should store sensitive data, such as credit card information, using encryption.
# """
text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services."
# 使用预训练的DistilBERT NER模型对文本进行处理
results = nlp(text)

print(results)
# 输出识别到的实体及其类型
for result in results:
    print(result["word"], result["entity"])