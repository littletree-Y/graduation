import nltk
from nltk.translate.bleu_score import SmoothingFunction

class SingleBleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, name='Bleu', portion=1, num_sentences=10):
        super().__init__()
        self.name = name
        self.test_data = test_text  # 这里test_text应该包含多行，为num_sentences倍数的real_text行数
        self.real_data = real_text  # real_text是参考数据，每行一个句子
        self.gram = gram
        self.sample_size = None  # 这里sample_size用不到，可以移除
        self.references = []  # References for the test data
        self.is_first = True  # to check if get_reference method has been called
        self.portion = portion  # 使用测试数据集的部分，默认使用全部
        self.num_sentences = num_sentences  # 每个real_text行生成的句子数

    def get_name(self):
        return self.name

    def get_score(self, ignore=False):
        if ignore:
            return 0

        if self.is_first:
            self.get_reference()
            self.is_first = False

        return self.get_bleu()

    def get_reference(self):
        # Read the real data lines, each line is a reference for the corresponding test texts
        with open(self.real_data, 'r') as real_file:
            self.references = [nltk.word_tokenize(line.strip()) for line in real_file]

    def get_bleu(self):
        weights = tuple((1. / self.gram for _ in range(self.gram)))
        bleu_scores = []
        n = self.num_sentences

        # Read the test data, for each reference there should be num_sentences of test
        with open(self.test_data, 'r') as test_file:
            test_lines = test_file.readlines()

         # Calculate the maximum number of test line chunks to match the number of references
        max_chunks = len(self.references)
        test_chunks = [test_lines[i: i + n] for i in range(0, max_chunks * n, n)]

        
        for test_chunk, ref in zip(test_chunks, self.references):
            # Tokenize each sentence in the chunk and compute the BLEU score against the single reference
            chunk_bleu_scores = [
                nltk.translate.bleu_score.sentence_bleu([ref], nltk.word_tokenize(test.strip()), weights, 
                                                        smoothing_function=SmoothingFunction().method1) 
                for test in test_chunk
            ]
            # Average BLEU score for this chunk
            bleu_scores.append(sum(chunk_bleu_scores) / n)

        # Average BLEU score across all chunks
        average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        return average_bleu