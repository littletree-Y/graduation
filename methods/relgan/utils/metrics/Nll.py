import numpy as np

from utils.metrics.Metrics import Metrics


class Nll(Metrics):
    def __init__(self, data_loader, pretrain_loss, x_real, sess, name='Nll', x_keywords=None, 
    x_keywords_len_list=None, x_fake_keywords=None, x_fake_keywords_len_list=None,is_keywords=False):
        super().__init__()
        self.name = name
        self.data_loader = data_loader
        self.sess = sess
        self.pretrain_loss = pretrain_loss
        self.x_real = x_real
        self.is_keywords = is_keywords
        self.x_keywords = x_keywords
        self.x_keywords_len_list = x_keywords_len_list
        self.x_fake_keywords = x_fake_keywords
        self.x_fake_keywords_len_list = x_fake_keywords_len_list

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.nll_loss()

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        for it in range(self.data_loader.num_batch):
            if self.is_keywords:
                batch, keywords, keywords_len_list = self.data_loader.next_batch()
                _, fake_keywords, fake_keywords_len_list = self.data_loader.random_batch()
                g_loss = self.sess.run(self.pretrain_loss, {self.x_real: batch,
                 self.x_keywords:keywords, self.x_keywords_len_list:keywords_len_list,
                 self.x_fake_keywords: fake_keywords, self.x_fake_keywords_len_list: fake_keywords_len_list})
            else:
                batch = self.data_loader.next_batch()
                g_loss = self.sess.run(self.pretrain_loss, {self.x_real: batch})
            nll.append(g_loss)
        return np.mean(nll)
