
"""
通用的模型还是无法识别安全实体 比如私人信息，密码
"""
import spacy

# python -m spacy download en_core_web_sm
if __name__ == "__main__":

    # 加载预训练的spaCy英文模型
    nlp = spacy.load("en_core_web_sm")

    # 示例功能需求文本
    # text = """
    # The system should allow users to securely access their personal information. 
    # Users should be able to authenticate using their email and password. 
    # The system should store sensitive data, such as credit card information, using encryption.
    # """
    text = "Apple is looking at buying U.K. startup for $1 billion"

    # 使用预训练的NER模型对文本进行处理
    doc = nlp(text)
    # print(doc)
    # 输出识别到的实体及其类型
    for ent in doc.ents:
        print(ent.text, ent.label_)