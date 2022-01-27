import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import seq2seq as s2s
import matplotlib.ticker as ticker
from tensorflow.keras.preprocessing import sequence

tf.keras.backend.clear_session()  # - for easy reset of notebook state

# check if GPU can be seen by TF
tf.config.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)  # only to check GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('gigaword32k.enc')

start = tokenizer.vocab_size + 1
end = tokenizer.vocab_size

BATCH_SIZE = 1
embedding_dim = 128
units = 256
vocab_size = end + 2
encoder = s2s.Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = s2s.Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = 'training_checkpoints-2020-Jun-30-09-26-31'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
chkpt_status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
chkpt_status.assert_existing_objects_matched()


def plot_attention(attention, article, summary):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='cividis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + article, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + summary, fontdict=fontdict)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))


art_max_len = 128
smry_max_len = 50


def greedy_search(article):
    attention_plot = np.zeros((smry_max_len, art_max_len))
    tokens = tokenizer.encode(article)
    if len(tokens) > art_max_len:
        tokens = tokens[:art_max_len]
    inputs = sequence.pad_sequences([tokens], padding='post', maxlen=art_max_len).squeeze()
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 0)
    summary = ''
    hidden = [tf.zeros((1, units)) for i in range(2)]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([start], 0)

    for t in range(smry_max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if predicted_id == end:
            return summary, article, attention_plot
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        summary += tokenizer.decode([predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)
    return summary, article, attention_plot


def summarize(article, algo='greedy'):
    if algo == 'greedy':
        summary, article, attention_plot = greedy_search(article)
    else:
        print("Algorithm {} not implemented".format(algo))
        return
    print('Input: %s' % article)
    print('** Predicted Summary: {}'.format(summary))
    attention_plot = attention_plot[:len(summary.split(' ')), :len(article.split(' '))]
    plot_attention(attention_plot, article.split(' '), summary.split(' '))


txt = "president georgi parvanov summoned france 's ambassador on wednesday in a show of displeasure over comments " \
      "from french president jacques chirac chiding east european nations for their support of washington on the " \
      "issue of iraq . "

summarize(txt.lower())


def beam_search(article, beam_width=3, art_max_len=128, sury_max_len=50, end_tk=end, verbose=True):
    attention_plot = np.zeros((smry_max_len, art_max_len))
    tokens = tokenizer.encode(article, )
    if len(tokens) > art_max_len:
        tokens = tokens[:art_max_len]
    inputs = sequence.pad_sequences([tokens], padding='post', maxlen=art_max_len).squeeze()
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 0)
    hidden = [tf.zeros((1, units)) for i in range(2)]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([start], 0)
    start_pt = [([start], dec_hidden, attention_plot, 0.0)]

    for t in range(smry_max_len):
        options = list()
        for row in start_pt:
            allend = True
            dec_input = row[0][-1]
            if dec_input != end_tk:
                dec_input = tf.expand_dims([dec_input], 0)
                dec_hidden = row[1]
                attn_plt = np.zeros((smry_max_len, art_max_len)) + row[2]
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

                attention_weights = tf.reshape(attention_weights, (-1,))
                attn_plt[t] = attention_weights.numpy()
                values, indices = tf.match.top_k(predictions[0], k=beam_width)
                for tokid, scre in zip(indices, values):
                    score = row[3] - np.log(scre)
                    options.append(row[0] + [tokid], dec_hidden, attn_plt, score)
                allend = False
            else:
                options.append(row)
        if allend:
            break
        start_pt = sorted(options, key=lambda tup: tup[3])[:beam_width]
    if verbose:
        for idx, row in enumerate(start_pt):
            tokens = [x for x in row[0] if x < end_tk]
            print('Summary {} with {:5f}: {}'.format(idx, row[3], tokenizer.decode(tokens)))
    summary = tokenizer.decode([x for x in start_pt[0][0] if x < end_tk])
    attention_plot = start_pt[0][2]
    return summary, article, attention_plot


def summarize(article, algo='greedy', beam_width=3, verbose=True):
    if algo == 'greedy':
        summary, article, attention_plot = greedy_search(article)
    elif algo == 'beam':
        summary, article, attention_plot = beam_search(article, beam_width=beam_width, verbose=verbose)
    else:
        print("Algorithm {} not implemented".format(algo))
        return
    print('Input: %s' % article)
    print('** Predicted Summary: {}'.format(summary))
    attention_plot = attention_plot[:len(summary.split(' ')), :len(article.split(' '))]
    plot_attention(attention_plot, article.split(' '), summary.split(' '))


def length_wu(step, score, alpha=0.):
    modifier = (((5 + step) ** alpha) / ((5 + 1) ** alpha))
    return (score / modifier)


def beam_search_norm(article, beam_width=3, art_max_len=128, smry_max_len=50, end_tk=end, alpha=0., verbose=True):
    attention_plot = np.zeros((smry_max_len, art_max_len))
    tokens = tokenizer.encode(article)
    if len(tokens) > art_max_len:
        tokens = tokens[:art_max_len]
    inputs = sequence.pad_sequences([tokens], padding='post', maxlen=art_max_len).squeeze()
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 0)
    hidden = [tf.zeros((1, units)) for i in range(2)]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([start], 0)
    start_pt = [([start], dec_hidden, attention_plot, 0.0)]
    for t in range(smry_max_len):
        options = list()
        for row in start_pt:
            allend = True
            dec_input = row[0][-1]
            if dec_input != end_tk:
                dec_input = tf.expand_dims([dec_input], 0)
                dec_hidden = row[1]
                attn_plt = np.zeros((smry_max_len, art_max_len)) + row[2]
                assert id(attn_plt) is not id(row[2])
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
                assert id(dec_hidden) is not id(row[1])
                attention_weights = tf.reshape(attention_weights, (-1,))
                attn_plt[t] = attention_weights.numpy()
                values, indices = tf.match.top_k(predictions[0], k=beam_width)
                for tokid, scre in zip(indices, values):
                    score = row[3] - np.log(scre)
                    score = length_wu(t, score, alpha)
                    options.append((row[0] + [tokid], dec_hidden, attn_plt, score))
                allend = False
            else:
                options.append(row)
        if allend:
            break
        start_pt = sorted(options, key=lambda tup: tup[3])[:beam_width]
    if verbose:
        for idx, row in enumerate(start_pt):
            tokens = [x for x in row[0] if x < end_tk]
            print('Summary {} with score {:5f}: {}'.format(idx, row[3], tokenizer.decode(tokens)))
    summary = tokenizer.decode([x for x in start_pt[0][0] if x < end_tk])
    attention_plot = start_pt[0][2]
    return summary, article, attention_plot


def summarize(article, algo='greedy', beam_width=3, alpha=0., verbose=True):
    if algo == 'greedy':
        summary, article, attention_plot = greedy_search(article)
    elif algo == 'beam':
        summary, article, attention_plot = beam_search(article, beam_width=beam_width, verbose=verbose)
    elif algo == 'beam-norm':
        summary, article, attention_plot = beam_search_norm(article, beam_width=beam_width, alpha=alpha,
                                                            verbose=verbose)
    else:
        print("Algorithm {} not implemented".format(algo))
        return
    print('Input: %s' % article)
    print('** Predicted Summary: {}'.format(summary))
    attention_plot = attention_plot[:len(summary.split(' ')), :len(article.split(' '))]
    plot_attention(attention_plot, article.split(' '), summary.split(' '))
