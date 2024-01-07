import tensorflow as tf
from utils.models.relational_memory import RelationalMemory
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from utils.ops import *


# The generator network based on the Relational Memory
def generator(x_real, keywords_onehot, keywords_len, temperature, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots, head_size, num_heads,
              hidden_dim, start_token):
    start_tokens = tf.constant([start_token] * batch_size, dtype=tf.int32)
    output_size = mem_slots * head_size * num_heads

    # build relation memory module
    g_embeddings = tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
    g_output_unit = create_output_unit(output_size, vocab_size)

    # Add keyword embedding
    mem_size = gen_mem._mem_size
    # 不同embedding
    # keyword_embeddings = tf.get_variable('g_keyword_emb', shape=[vocab_size, mem_size],
    #                                      initializer=create_linear_initializer(vocab_size))
    # emb_keywords_re = tf.matmul(tf.reshape(keywords_onehot, [-1, vocab_size]), keyword_embeddings)
    # emb_keywords = tf.reshape(emb_keywords_re, [batch_size, -1, mem_size])  # batch_size x num_keywords x mem_size

    # 相同embedding
    emb_keywords_re = tf.matmul(tf.reshape(keywords_onehot, [-1, vocab_size]), g_embeddings)
    emb_keywords = tf.reshape(emb_keywords_re, [batch_size, -1, gen_emb_dim])   # batch_size x num_keywords x gen_emb_dim
    # Map the keyword embeddings to the generator's memory size
    keyword_fc = tf.get_variable('keywords_fc', shape=[gen_emb_dim, mem_size],
                                initializer=create_linear_initializer(gen_emb_dim))
    emb_keywords = tf.tensordot(emb_keywords, keyword_fc, axes=[[2], [0]])  # batch_size x num_keywords x mem_size


    # Compute a weighted average of keyword embeddings
    keyword_lengths = tf.expand_dims(tf.cast(keywords_len, tf.float32), -1)  # batch_size x 1
    weights = 1.0 / keyword_lengths  # batch_size x 1
    weights = tf.tile(weights, [1, tf.shape(emb_keywords)[1]])  # batch_size x num_keywords

    # Create a mask for existing keywords based on keywords_len
    keyword_mask = tf.sequence_mask(keywords_len, maxlen=tf.shape(emb_keywords)[1], dtype=tf.float32)  # batch_size x num_keywords
    weights = weights * keyword_mask  # batch_size x num_keywords

    weights = tf.expand_dims(weights, -1)  # batch_size x num_keywords x 1
    # weights = tf.tile(weights, [1, 1, mem_size])  # batch_size x num_keywords x mem_size
    emb_keywords = tf.reduce_sum(weights * emb_keywords, axis=1)  # batch_size x mem_size

    emb_keywords = tf.reshape(emb_keywords, [batch_size, 1, mem_size])  # batch_size x 1 x mem_size
    emb_keywords_tiled = tf.tile(emb_keywords, [1, mem_slots, 1])  # batch_size x mem_slots x mem_size


    # initial states
    init_states = gen_mem.initial_state(batch_size) # (batch_size, self._mem_slots, self._mem_size)

    # Add keyword memory to initial states
    init_states = init_states + emb_keywords_tiled

    # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
    gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)
    gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                    infer_shape=True)  # generator output (relaxed of gen_x)

    # the generator recurrent module used for adversarial training
    def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)  # hidden_memory_tuple
        o_t = g_output_unit(mem_o_t)  # batch x vocab, logits not probs
        gumbel_t = add_gumbel(o_t)
        next_token = tf.stop_gradient(tf.argmax(gumbel_t, axis=1, output_type=tf.int32))
        next_token_onehot = tf.one_hot(next_token, vocab_size, 1.0, 0.0)

        x_onehot_appr = tf.nn.softmax(tf.multiply(gumbel_t, temperature))  # one-hot-like, [batch_size x vocab_size]

        # x_tp1 = tf.matmul(x_onehot_appr, g_embeddings)  # approximated embeddings, [batch_size x emb_dim]
        x_tp1 = tf.nn.embedding_lookup(g_embeddings, next_token)  # embeddings, [batch_size x emb_dim]

        gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(next_token_onehot, x_onehot_appr), 1))  # [batch_size], prob
        gen_x = gen_x.write(i, next_token)  # indices, [batch_size]

        gen_x_onehot_adv = gen_x_onehot_adv.write(i, x_onehot_appr)

        return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv

    # build a graph for outputting sequential tokens
    _, _, _, gen_o, gen_x, gen_x_onehot_adv = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3, _4, _5: i < seq_len,
        body=_gen_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                   init_states, gen_o, gen_x, gen_x_onehot_adv))

    gen_o = tf.transpose(gen_o.stack(), perm=[1, 0])  # batch_size x seq_len
    gen_x = tf.transpose(gen_x.stack(), perm=[1, 0])  # batch_size x seq_len

    gen_x_onehot_adv = tf.transpose(gen_x_onehot_adv.stack(), perm=[1, 0, 2])  # batch_size x seq_len x vocab_size

    # ----------- pre-training for generator -----------------
    x_emb = tf.transpose(tf.nn.embedding_lookup(g_embeddings, x_real), perm=[1, 0, 2])  # seq_len x batch_size x emb_dim
    g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)

    ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len)
    ta_emb_x = ta_emb_x.unstack(x_emb)

    # the generator recurrent moddule used for pre-training
    def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
        mem_o_t, h_t = gen_mem(x_t, h_tm1)
        o_t = g_output_unit(mem_o_t)
        g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch_size x vocab_size
        x_tp1 = ta_emb_x.read(i)
        return i + 1, x_tp1, h_t, g_predictions

    # build a graph for outputting sequential tokens
    _, _, _, g_predictions = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3: i < seq_len,
        body=_pretrain_recurrence,
        loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                   init_states, g_predictions))

    g_predictions = tf.transpose(g_predictions.stack(),
                                 perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

    # pre-training loss
    pretrain_loss = -tf.reduce_sum(
        tf.one_hot(tf.to_int32(tf.reshape(x_real, [-1])), vocab_size, 1.0, 0.0) * tf.log(
            tf.clip_by_value(tf.reshape(g_predictions, [-1, vocab_size]), 1e-20, 1.0)
        )
    ) / (seq_len * batch_size)


    # pretain keywords loss
    # 平均embedding
    # Calculate the keyword loss
    # generated_text_embedding = tf.matmul(tf.reshape(g_predictions, [-1, vocab_size]), g_embeddings)  # (batch_size * seq_len) x emb_dim
    # generated_text_embedding = tf.reshape(generated_text_embedding, [batch_size, seq_len, gen_emb_dim])  # batch_size x seq_len x emb_dim
    # generated_text_embedding_mean = tf.reduce_mean(generated_text_embedding, axis=1)  # batch_size x emb_dim

    # # Calculate target_keywords_embedding
    # keywords_onehot_flat = tf.reshape(tf.cast(keywords_onehot, tf.float32), [-1, vocab_size])  # (batch_size * num_keywords) x vocab_size
    # target_keywords_embedding_flat = tf.matmul(keywords_onehot_flat, g_embeddings)  # (batch_size * num_keywords) x emb_dim
    # target_keywords_embedding = tf.reshape(target_keywords_embedding_flat, [batch_size, -1, gen_emb_dim])  # batch_size x num_keywords x emb_dim
    # target_keywords_embedding_mean = tf.reduce_mean(target_keywords_embedding, axis=1)  # batch_size x emb_dim

    # # Calculate the cosine similarity between the generated text and the target keywords
    # similarity = cosine_similarity(generated_text_embedding_mean, target_keywords_embedding_mean)

    # # Define the keyword loss as the negative cosine similarity
    # pretrain_keyword_loss = -tf.reduce_mean(similarity)

    # # keyword loss for adversarial training
    # generated_text_adv = tf.matmul(tf.reshape(gen_x_onehot_adv, [-1, vocab_size]), g_embeddings)  # (batch_size * seq_len) x emb_dim
    # generated_text_adv_embedding = tf.reshape(generated_text_adv, [batch_size, seq_len, gen_emb_dim])  # batch_size x seq_len x emb_dim

    # # Calculate the cosine similarity between the generated text and the target keywords for adversarial training
    # similarity_adv = cosine_similarity(tf.reduce_mean(generated_text_adv_embedding, axis=1), target_keywords_embedding_mean)

    # # Define the keyword loss as the negative cosine similarity for adversarial training
    # adv_keyword_loss = -tf.reduce_mean(similarity_adv)


    generated_text_embedding = tf.matmul(tf.reshape(g_predictions, [-1, vocab_size]), g_embeddings)  # (batch_size * seq_len) x emb_dim
    keywords_onehot_flat = tf.reshape(tf.cast(keywords_onehot, tf.float32), [-1, vocab_size])  # (batch_size * num_keywords) x vocab_size
    target_keywords_embedding = tf.matmul(keywords_onehot_flat, g_embeddings)  # (batch_size * num_keywords) x emb_dim
    # target_keywords_embedding = tf.reshape(target_keywords_embedding, [-1, gen_emb_dim])  # (batch_size * num_keywords) x emb_dim

     # Convert target_keywords_embedding to a RaggedTensor
    # target_keywords_embedding_ragged = tf.RaggedTensor.from_row_lengths(target_keywords_embedding, row_lengths=keywords_len)

    generated_text_adv_embedding = tf.matmul(tf.reshape(gen_x_onehot_adv, [-1, vocab_size]), g_embeddings)  # (batch_size * seq_len) x emb_dim

    def compute_keyword_loss(args):
        generated_text_embedding_sample, target_keywords_embedding_sample, num_keywords_sample = args
        
        # 确保target_keywords_embedding_sample是二维张量
        target_keywords_embedding_sample = tf.reshape(target_keywords_embedding_sample, 
                                                    [-1, tf.shape(generated_text_embedding_sample)[-1]])
        
        # 截取前num_keywords_sample个关键字嵌入
        def slice_keywords(keywords_embeddings, count):
            return keywords_embeddings[:count, :]
        
        valid_keywords_embeddings_sample = tf.cond(
            num_keywords_sample > 0,
            lambda: slice_keywords(target_keywords_embedding_sample, num_keywords_sample),
            lambda: tf.zeros([0, tf.shape(generated_text_embedding_sample)[-1]])
        )
        
        # 计算生成文本和有效关键词嵌入之间的余弦相似度
        similarity = cosine_similarity(generated_text_embedding_sample, valid_keywords_embeddings_sample)
        
        # 对于每个生成的单词，找到与其最相似的关键词的相似度
        max_similarity = tf.reduce_max(similarity, axis=1)
        
        # 自定义损失：取相似度最大值的负数作为损失
        keyword_loss = -tf.reduce_sum(max_similarity)
        return keyword_loss

    #  Compute the keyword loss for each element in the batch
    elems = (generated_text_embedding, target_keywords_embedding, keywords_len)
    pretrain_keyword_loss = tf.map_fn(
        compute_keyword_loss, 
        elems, 
        dtype=tf.float32, 
        back_prop=True
    )

    pretrain_keyword_loss = tf.reduce_mean(pretrain_keyword_loss)

    # Calculate keyword loss for each sample in the batch for adversarial training
    adv_keyword_loss = tf.map_fn(compute_keyword_loss, elems=(generated_text_adv_embedding, target_keywords_embedding, keywords_len), dtype=tf.float32, swap_memory=True)
    adv_keyword_loss = tf.reduce_mean(adv_keyword_loss)



    return gen_x_onehot_adv, gen_x, pretrain_loss, gen_o, pretrain_keyword_loss, adv_keyword_loss


# The discriminator network based on the CNN classifier
def discriminator(x_onehot, keywords_onehot, keywords_len, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn):
    # get the embedding dimension for each presentation
    emb_dim_single = int(dis_emb_dim / num_rep)
    assert isinstance(emb_dim_single, int) and emb_dim_single > 0

    filter_sizes = [2, 3, 4, 5]
    num_filters = [300, 300, 300, 300]
    dropout_keep_prob = 0.75

    d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                   initializer=create_linear_initializer(vocab_size))
    input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
    emb_x_re = tf.matmul(input_x_re, d_embeddings) # (batch x seq_len) x vocab_size  vocab_size x dis_emb_dim
    # emb_x_re  = (batch x seq_len) x dis_emb_dim
    emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])  # batch_size x seq_len x dis_emb_dim

    # Add keyword embedding
    keyword_embeddings = tf.get_variable('d_keyword_emb', shape=[vocab_size, dis_emb_dim],
                                         initializer=create_linear_initializer(vocab_size))
    emb_keywords_re = tf.matmul(tf.reshape(keywords_onehot, [-1, vocab_size]), keyword_embeddings)
    emb_keywords = tf.reshape(emb_keywords_re, [batch_size, -1, dis_emb_dim])  # batch_size x num_keywords x dis_emb_dim

    # Compute a weighted average of keyword embeddings
    keyword_lengths = tf.expand_dims(tf.cast(keywords_len, tf.float32), -1)  # batch_size x 1
    weights = 1.0 / keyword_lengths  # batch_size x 1
    weights = tf.tile(weights, [1, tf.shape(emb_keywords)[1]])  # batch_size x num_keywords

    # Create a mask for existing keywords based on keywords_len
    keyword_mask = tf.sequence_mask(keywords_len, maxlen=tf.shape(emb_keywords)[1], dtype=tf.float32)  # batch_size x num_keywords
    weights = weights * keyword_mask  # batch_size x num_keywords

    weights = tf.expand_dims(weights, -1)  # batch_size x num_keywords x 1
    emb_keywords = tf.reduce_sum(weights * emb_keywords, axis=1)  # batch_size x dis_emb_dim

    emb_keywords = tf.reshape(emb_keywords, [batch_size, 1, dis_emb_dim])  # batch_size x 1 x dis_emb_dim
    emb_keywords_tiled = tf.tile(emb_keywords, [1, seq_len, 1])  # batch_size x seq_len x dis_emb_dim

    emb_x = emb_x + emb_keywords_tiled  # Add keyword embedding to each time step

    emb_x_expanded = tf.expand_dims(emb_x, -1)  # batch_size x seq_len x dis_emb_dim x 1
    print('shape of emb_x_expanded: {}'.format(emb_x_expanded.get_shape().as_list()))

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for filter_size, num_filter in zip(filter_sizes, num_filters):
        conv = conv2d(emb_x_expanded, num_filter, k_h=filter_size, k_w=emb_dim_single,
                      d_h=1, d_w=emb_dim_single, sn=sn, stddev=None, padding='VALID',
                      scope="conv-%s" % filter_size)  # batch_size x (seq_len-k_h+1) x num_rep x num_filter
        out = tf.nn.relu(conv, name="relu")
        pooled = tf.nn.max_pool(out, ksize=[1, seq_len - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1], padding='VALID',
                                name="pool")  # batch_size x 1 x num_rep x num_filter
        pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = sum(num_filters)
    h_pool = tf.concat(pooled_outputs, 3)  # batch_size x 1 x num_rep x num_filters_total
    print('shape of h_pool: {}'.format(h_pool.get_shape().as_list()))
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add highway
    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)  # (batch_size*num_rep) x num_filters_total

    # Add dropout
    h_drop = tf.nn.dropout(h_highway, dropout_keep_prob, name='dropout')

    # fc
    fc_out = linear(h_drop, output_size=100, use_bias=True, sn=sn, scope='fc')
    logits = linear(fc_out, output_size=1, use_bias=True, sn=sn, scope='logits')
    logits = tf.squeeze(logits, -1)  # batch_size*num_rep

    return logits


def cosine_similarity(a, b):
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=1))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=1))
    a_norm = tf.where(tf.equal(a_norm, 0), tf.ones_like(a_norm), a_norm)
    b_norm = tf.where(tf.equal(b_norm, 0), tf.ones_like(b_norm), b_norm)
    normalize_a = a / tf.expand_dims(a_norm, -1)
    normalize_b = b / tf.expand_dims(b_norm, -1)
    return tf.matmul(normalize_a, tf.transpose(normalize_b))
