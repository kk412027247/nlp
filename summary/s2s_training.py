import tensorflow as tf
import tensorflow_datasets as tfds
import os
import seq2seq as s2s
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
import argparse
import datetime
import time

parser = argparse.ArgumentParser(description='Train the seq2seq model')
parser.add_argument('--checkpoint', type=str, help='Name of the checkpoint directory to restart training from.')

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
            target_vocab_size=2 ** 15)
        tokenizer.save_to_file(file)
    print('Tokenizer ready. Total vocabulary size:', tokenizer.vocab_size)
    return tokenizer


@tf.function
def train_step(inp, targ, enc_hidden, max_gradient_norm=5):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([start] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += s2s.loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    optimizer.apply_gradients(zip(clipped_gradients, variables))
    return batch_loss


if __name__ == '__main__':
    args = parser.parse_args()
    setupGPU()
    ds_train, _, _ = load_data()
    tokenizer = get_tokenizer(ds_train)
    txt = 'Coronavirus spread surprised everyone'
    print(txt, '=>', tokenizer.encode(txt.lower()))
    for ts in tokenizer.encode(txt.lower()):
        print('{} ----> {}'.format(ts, tokenizer.decode([ts])))
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

    steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 128
    units = 256
    EPOCHS = 6

    encoder = s2s.Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
    decoder = s2s.Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

    lr_schedule = InverseTimeDecay(0.001, decay_steps=steps_per_epoch * (EPOCHS / 2), decay_rate=2, staircase=False)
    optimizer = Adam(lr_schedule)

    if args.checkpoint is None:
        dt = datetime.datetime.today().strftime('%Y-%b-%d-%H-%M-%S')
        checkpoint_dir = './training_checkpoints-' + dt
    else:
        checkpoint_dir = args.checkpoint
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    if args.checkpoint is not None:
        print('checkpoint being restored: ', tf.train.latest_checkpoint(checkpoint_dir))
        chkpt_status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        chkpt_status.assert_existing_objects_matched()
    else:
        print('string new training run from scratch')
    print('new checkpoints will be stored in: ', checkpoint_dir)
    print('starting training. total number of steps/epoch:', steps_per_epoch)

    for epoch in range(EPOCHS):
        start_tm = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (art, smry)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(art, smry, enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                ts = datetime.datetime.now().strftime('%d-%b-%Y (%H:%M:%S)')
                print('[{}] Epoch {} Batch {} Loss {:.6f}'.format(ts, epoch + 1, batch, batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.6}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time take for 1 eopch {} sec\n'.format(time.time() - start_tm))
    encoder.summary()
    decoder.summary()
