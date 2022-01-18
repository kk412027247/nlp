import tensorflow as tf
import tensorflow_datasets as tfds
import os
import seq2seq as s2s

proxy = 'http://127.0.0.1:8001'
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['TFDS_HTTPS_PROXY'] = proxy


def load_data():
    print(" Loading the dataset")
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'gigaword',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_val, ds_test


def setupGPU():
    # chck if GPU can be seen by TF
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
            print(e)


def get_tokenizer(data, file='gigaword32k.enc'):
    if os.path.exists(file + '.subwords'):
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(file)
    else:
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            ((art.numpy() + b' ' + smm.numpy()) for art, smm in data),
            target_vocab_size=2 ** 15
        )
        tokenizer.save_to_file(file)
    print('Tokenizer ready. Total vocabulary size:', tokenizer.vocab_size)
    return tokenizer


if __name__ == '__main__':
    setupGPU()
    ds_train, _, _ = load_data()
    tokenizer = get_tokenizer(ds_train)
    txt = 'Coronavirus spread surprised everyone'
    print(txt, '=>', tokenizer.encode(txt.lower()))
    for ts in tokenizer.encode(txt.lower()):
        print('{}---->{}'.format(ts, tokenizer.decode([ts])))
    start = tokenizer.vocab_size + 1
    end = tokenizer.vocab_size
    vocab_size = end + 2
    BUFFER_SIZE = 3500000
    BATCH_SIZE = 64
    train = ds_train.take(BUFFER_SIZE)
    print('Dataset sample taken')
    train_dataset = train.map(s2s.tf_encode)
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    print('Dataset batching done')
