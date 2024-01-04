import numpy as np
import random


class RealDataKeywordsLoader():
    def __init__(self, batch_size, seq_length, keywords_len, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token
        self.keywords_len = keywords_len
        self.keywords_stream = []
        self.keywords_len_list = []
        self.batch_keywords_len = []

    def create_batches(self, data_file, oracle_keywords_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)
        
        with open(oracle_keywords_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.keywords_len:
                    self.keywords_stream.append(parse_line[:self.keywords_len])
                    self.keywords_len_list.append(self.keywords_len)
                else:
                    self.keywords_len_list.append(len(parse_line))
                    while len(parse_line) < self.keywords_len:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.keywords_len:
                        self.keywords_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.keywords_stream = self.keywords_stream[:self.num_batch * self.batch_size]
        self.keywords_len_list = self.keywords_len_list[:self.num_batch * self.batch_size]
        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.keywords_batches = np.split(np.array(self.keywords_stream), self.num_batch, 0)
        self.batch_keywords_len = np.split(np.array(self.keywords_len_list), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        keywords = self.keywords_batches[self.pointer]
        keywords_len_list = self.batch_keywords_len[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, keywords, keywords_len_list

    def random_batch(self):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        keywords = self.keywords_batches[rn_pointer]
        keywords_len_list = self.batch_keywords_len[rn_pointer]
        return ret, keywords, keywords_len_list

    def reset_pointer(self):
        self.pointer = 0


class TestRealDataKeywordsLoader():
    def __init__(self, batch_size, seq_length, keywords_len, end_token=0, repeat_times=1):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token
        self.keywords_len = keywords_len
        self.keywords_stream = []
        self.keywords_len_list = []
        self.batch_keywords_len = []
        self.repeat_times = repeat_times

    def create_batches(self, data_file, oracle_keywords_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.extend([parse_line[:self.seq_length]] * self.repeat_times)
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.extend([parse_line] * self.repeat_times)

        with open(oracle_keywords_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.keywords_len:
                    self.keywords_stream.extend([parse_line[:self.keywords_len]] * self.repeat_times)
                    self.keywords_len_list.extend([self.keywords_len] * self.repeat_times)
                else:
                    self.keywords_len_list.extend([len(parse_line)] * self.repeat_times)
                    while len(parse_line) < self.keywords_len:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.keywords_len:
                        self.keywords_stream.extend([parse_line] * self.repeat_times)

         # If the last batch is smaller than the batch size, fill it with the last data
        if len(self.token_stream) % self.batch_size != 0:
            last_data_token = [self.token_stream[-1]] * (self.batch_size - len(self.token_stream) % self.batch_size)
            last_data_keywords = [self.keywords_stream[-1]] * (self.batch_size - len(self.keywords_stream) % self.batch_size)
            last_data_keywords_len = [self.keywords_len_list[-1]] * (self.batch_size - len(self.keywords_len_list) % self.batch_size)
            self.token_stream.extend(last_data_token)
            self.keywords_stream.extend(last_data_keywords)
            self.keywords_len_list.extend(last_data_keywords_len)

        self.num_batch = int(len(self.token_stream) / self.batch_size)

        self.sequence_batches = np.split(np.array(self.token_stream), self.num_batch)
        self.keywords_batches = np.split(np.array(self.keywords_stream), self.num_batch)
        self.batch_keywords_len = np.split(np.array(self.keywords_len_list), self.num_batch)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batches[self.pointer]
        keywords = self.keywords_batches[self.pointer]
        keywords_len_list = self.batch_keywords_len[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, keywords, keywords_len_list

    def random_batch(self):
        rn_pointer = random.randint(0, self.num_batch - 1)
        ret = self.sequence_batches[rn_pointer]
        keywords = self.keywords_batches[rn_pointer]
        keywords_len_list = self.batch_keywords_len[rn_pointer]
        return ret, keywords, keywords_len_list

    def reset_pointer(self):
        self.pointer = 0