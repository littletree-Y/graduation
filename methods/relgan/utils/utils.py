import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pprint
from utils.text_process import *

pp = pprint.PrettyPrinter()


def generate_samples(sess, gen_x, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(sess.run(gen_x))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for sent in generated_samples:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(sent)
        return np.array(codes)
    codes = ""
    for sent in generated_samples:
        buffer = ' '.join([str(x) for x in sent]) + '\n'
        codes += buffer
    return codes

def generate_samples_keywords(sess, gen_x, test_loader, x_keywords, x_keywords_len_list,output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    test_loader.reset_pointer()
    for it in range(test_loader.num_batch):
        _, keywords, keywords_len_list = test_loader.next_batch()
        generated_samples.extend(sess.run(gen_x, feed_dict={x_keywords:keywords, x_keywords_len_list:keywords_len_list}))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for sent in generated_samples:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(sent)
        return np.array(codes)
    codes = ""
    for sent in generated_samples:
        buffer = ' '.join([str(x) for x in sent]) + '\n'
        codes += buffer
    return codes


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, g_pretrain_op, g_pretrain_loss, x_real, x_keywords, x_keywords_len_list,data_loader, is_keywords=False):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        if is_keywords:
            batch, keywords, keywords_len_list = data_loader.next_batch()
            _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: batch, x_keywords:keywords, x_keywords_len_list:keywords_len_list })
        else:
            batch = data_loader.next_batch()
            _, g_loss = sess.run([g_pretrain_op, g_pretrain_loss], feed_dict={x_real: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def plot_csv(csv_file, pre_epoch_num, metrics, method):
    names = [str(i) for i in range(len(metrics) + 1)]
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=0, skip_footer=0, names=names)
    for idx in range(len(metrics)):
        metric_name = metrics[idx].get_name()
        plt.figure()
        plt.plot(data[names[0]], data[names[idx + 1]], color='r', label=method)
        plt.axvline(x=pre_epoch_num, color='k', linestyle='--')
        plt.xlabel('training epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plot_file = os.path.join(os.path.dirname(csv_file), '{}_{}.pdf'.format(method, metric_name))
        print(plot_file)
        plt.savefig(plot_file)


def get_oracle_file(data_file, oracle_file, seq_len):
    tokens = get_tokenlized(data_file)
    word_set = get_word_list(tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)
    with open(oracle_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return index_word_dict

def get_oracle_file_keywords(data_file, keywords_file, oracle_file, oracle_keywords_file, 
    seq_len, keywords_len, test_file, oracle_test_file, test_keywords_file, oracle_test_keywords_file):
    tokens = get_tokenlized(data_file)
    # add test tokens
    test_tokens = get_tokenlized(test_file)
    word_set = get_word_list(tokens + test_tokens)

    [word_index_dict, index_word_dict] = get_dict(word_set)
    with open(oracle_file, 'w') as outfile:
        outfile.write(text_to_code(tokens, word_index_dict, seq_len))
    
    keywords_tokens = get_tokenlized(keywords_file)
    with open(oracle_keywords_file, 'w') as outfile:
        outfile.write(text_to_code(keywords_tokens, word_index_dict, keywords_len))

    test_tokens = get_tokenlized(test_file)
    with open(oracle_test_file, 'w') as outfile:
        outfile.write(text_to_code(test_tokens, word_index_dict, keywords_len))

    test_keywords_tokens = get_tokenlized(test_keywords_file)
    with open(oracle_test_keywords_file, 'w') as outfile:
        outfile.write(text_to_code(test_keywords_tokens, word_index_dict, keywords_len))
    return index_word_dict

def get_real_test_file(generator_file, gen_save_file, iw_dict):
    codes = get_tokenlized(generator_file)
    with open(gen_save_file, 'w') as outfile:
        outfile.write(code_to_text(codes=codes, dictionary=iw_dict))
