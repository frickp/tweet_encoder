"""
The goal of this module is to learn sentence encoding for tweets

Current encoder strategies implemented:
  - embedding maxpool (baseline)
  - LSTM-based embedding autoencoder
  - LSTM, maxpool-based sentiment prediction encoder

Each technique is implemented as its own class, which are each derived from
a parent Encoder class. This parent class handles methods and attributes common to
all networks, including data input, storage, model training, etc.

The model builds tensorflow models, and relies on some preprocessing scripts
from keras.

Future ideas:
last-slice hidden state for bi-LSTM in prediction network
  - I think part of poor performance of maxpooling in LSTM prediction network
    is that it does not give stronger weight to the end of the sequence, after
    all information has been seen

"""

import pickle
import re
import tensorflow as tf
import numpy as np
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_glove_model(glove_file):
    """load pre-trained glove vector into memory as a dictionary"""
    print("Loading Glove Model")

    f = open(glove_file, 'r')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = [float(val) for val in split_line[1:]]
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

def standardize_html(tweet):
    """replace all http links with 'URL'"""
    return re.sub(r"http\S+", "URL", tweet)

def standardize_www(tweet):
    """replace all www links with 'URL'"""
    return re.sub(r"www\S+", "URL", tweet)

def standardize_mentions(tweet):
    """replace all mentions with 'twittermention'"""
    return re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z_]+[A-Za-z0-9_]+)",\
                  "twittermention", tweet)

def query_most_similar_twts(model, tree, k, len_embed, index=None):
    """
    For a given trained SentenceEncoder model, and a trained Ball Tree,
    return the `k`- most similar tweets.

    model: pre-trained SentenceEncoder model
    tree: Ball Tree object trained on learned embeddings
    k: number of most similar tweets to print
    index: index location(`.iloc`) of tweet. If None, choose random index location
    len_embed: total number of tweet embeddings.
               equal to `model.df_test.shape[0]`
    """
    if not index:
        index = np.random.randint(0, len_embed)
    _, sim_idxs = tree.query([model.test_set_embeddings[index]], k=k)
    print('INITIAL TWEET\n{}\t{}\t{}\n'.format(
        index, model.df_test.iloc[sim_idxs[0][0]]['Sentiment'],
        model.df_test['normed_text'].iloc[sim_idxs[0][0]]))
    print('MOST SIMILAR TWEETS (index, sentiment, tweet):')
    #sim_idxs[0]
    for sim_ix in sim_idxs[0][1:]:
        print('{}\t{}\t{}'.format(sim_ix,
                                  model.df_test.iloc[sim_ix]['Sentiment'],
                                  model.df_test['normed_text'].iloc[sim_ix]))



class Encoder:
    """
    Parent class for all subsequent encoder models. Stores parameters and initialiazes
    attributes used for data storage

    initialization parameters:
    glove_vec_size: dimensions (1D) of word embeddings used
    top_n_words_to_process: specify how many words to keep for subsequent learning
    glove_path: `/path/to/stored/embeddings`
    labeled_twts: `/path/to/tweets`
    max_seq_length: max number of words per document – zero-pad the rest
    num_total_twts: number of documents in dataset
    max_seq_length: maximum number of words per document
    lstm_units: number of hidden units to use for LSTM
    emb_size: dimension of word embedding vectors
    """
    def __init__(self, glove_vec_size, top_n_words_to_process, glove_path, labeled_twts,
                 num_total_twts, max_seq_length, batch_size, lstm_units, emb_size):
        self.glove_vec_size = glove_vec_size
        self.top_n_words_to_process = top_n_words_to_process
        self.glove_path = glove_path
        self.labeled_twts = labeled_twts
        self.num_total_twts = num_total_twts
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.emb_size = emb_size

        # store needed data as attributes
        self.df_train = None
        self.df_test = None
        self.training_seq = None
        self.testing_seq = None
        self.training_labels = None
        self.testing_labels = None
        self.training_texts = None
        self.testing_texts = None
        self.word_index = None
        self.normed_embedding_mat = None

        # attributes to store relevant model iterators and layers
        self.optimizer = None
        self.saver = None
        self.enc_flat = None
        self.iterator = None
        self.train_init_op = None
        self.test_init_op = None
        self.loss = None
        self.test_set_embeddings = None
        self.training_loss = None

    def read_data(self):
        """
        Read data from file, process texts, and store training data as array of cleaned strings
        """
        df = pd.read_csv(self.labeled_twts,
                         error_bad_lines=False)  # labeled data
        df.reset_index(drop=True, inplace=True)
        df_sub = df.sample(n=self.num_total_twts, random_state=42)
        df_sub['normed_text'] = df_sub['SentimentText']\
            .apply(standardize_mentions)\
            .apply(standardize_html)\
            .apply(standardize_www)
        df_train, df_test = train_test_split(
            df_sub, test_size=0.1, random_state=42)
        self.training_texts = df_train['normed_text'].values
        self.testing_texts = df_test['normed_text'].values
        training_labels = to_categorical(df_train['Sentiment'].values)
        testing_labels = to_categorical(df_test['Sentiment'].values)

        self.df_train = df_train
        self.df_test = df_test
        self.training_labels = training_labels
        self.testing_labels = testing_labels

    def text_preprocessing(self):
        """
        Use keras functions to preprocess text and output resultant padded sequences
        """
        tokenizer = Tokenizer(num_words=self.top_n_words_to_process)
        tokenizer.fit_on_texts(self.training_texts)
        raw_train_sequences = tokenizer.texts_to_sequences(self.training_texts)
        raw_test_sequences = tokenizer.texts_to_sequences(self.testing_texts)

        self.word_index = tokenizer.word_index
        self.training_seq = pad_sequences(
            raw_train_sequences, maxlen=self.max_seq_length)
        self.testing_seq = pad_sequences(
            raw_test_sequences, maxlen=self.max_seq_length)

    def build_embedding_matrix(self, mat_name_save=None):
        """
        load pretrained glove embeddings, standardize the embeddings, and optionally save them
        """
        glove_word_vecs = load_glove_model(self.glove_path)
        embedding_matrix = np.zeros(
            (len(self.word_index) + 1, self.glove_vec_size),
            dtype=np.float32)
        for word, i in self.word_index.items():
            embedding_vector = glove_word_vecs.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_mean = np.mean(embedding_matrix, axis=0)
        embedding_mean_centered = embedding_matrix - embedding_mean
        embedding_stdev = np.std(embedding_mean_centered, axis=0)
        normed_embedding_mat = np.divide(
            embedding_mean_centered, embedding_stdev)
        #self.normed_embedding_mat = normed_embedding_mat
        if mat_name_save is not None:
            with open(mat_name_save, 'wb') as f:
                pickle.dump(normed_embedding_mat, f)

    def load_embedding_matrix(self, pickle_file):
        """
        Load embedding matrix, if already computed. This is an n x m matrix, 
        where `m` is the number + 1 of
        top words kept and `n` is the dimension of word vectors used
        """
        with open(pickle_file, 'rb') as f:
            normed_embedding_mat = pickle.load(f)
        self.normed_embedding_mat = normed_embedding_mat

    def forward_prop_test_set(self, ckpt_path='.', num_epochs=2000):
        """
        Forward propagate test data through the model
        
        ckpt_path: path/to/*ckpt which is saved model object to read in
        num_epochs: number of epochs to run forward prop on. 
                    Default 2000 (2000 x 64 = 128_000)
        """
        test_set_embeddings = np.zeros(
            [self.batch_size * num_epochs, self.enc_flat.shape[1]])
        with tf.Session() as sess:
          # Restore variables from disk.
            self.saver.restore(sess, ckpt_path)
            print("Model restored.")
            sess.run([self.test_init_op])
            for epoch in range(num_epochs):
                emb_temp = sess.run([self.enc_flat])[0]
                test_set_embeddings[epoch *
                                    self.batch_size:(epoch+1)*self.batch_size] = emb_temp
        test_set_embeddings = test_set_embeddings[0:self.df_test.shape[0], :]
        self.test_set_embeddings = test_set_embeddings
    
    def train_model(self, num_epochs=1000, epoch_print_intvl=100, save_path=None):
        """
        Train the model based on graph defined in the respective chile class

        num_epochs: Number of epochs to train model for
        epoch_print_intvl: interval epoch when loss is displayed
        save_path: location to save derived objects
        """
        self.training_loss = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([self.train_init_op])
            for epoch in range(num_epochs):
                _, l = sess.run([self.optimizer, self.loss])
                self.training_loss.append(l)
                if epoch % epoch_print_intvl == 0:
                    print('loss at epoch {}: {}'.format(epoch, l))
            if save_path is not None:
                self.saver.save(sess, save_path+'.ckpt')
                with open(save_path + '_training_losses', 'wb') as f:
                    pickle.dump(self.training_loss, f)


class EmbeddingAutoencoder(Encoder):
    """
    Class for learning encodings using LSTM network. Strategy is to reduce dimensionality
    using LSTM, extract encodings, then expand the dimensions. Optimize similarities, by
    cosine similarity, of initial embedding layer and final embedding layer
    """
    def __init__(self, glove_vec_size, top_n_words_to_process,\
                 glove_path, labeled_twts, emb_size, lstm_units,\
                 num_total_twts, max_seq_length, batch_size):
        """
        Set attributes to be used in subsequent functions and stored output
        """
        Encoder.__init__(self, glove_vec_size, top_n_words_to_process, glove_path, labeled_twts,
                         num_total_twts, max_seq_length, batch_size, lstm_units, emb_size)
       
    def build_embedding_autoencoder_graph(self):
        """
        Define graph for using LSTM to define encoding
        Use gradient descent to optimize for cosine similarity
        of the decoded embedding layer and the original embeddings
        """
        tf.reset_default_graph()
        train_data = (self.training_seq, self.training_labels)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)\
                                       .repeat().batch(self.batch_size)
        self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                        train_dataset.output_shapes)

        test_data = (self.testing_seq, self.testing_labels)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data).repeat().batch(self.batch_size)

        features, _ = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(train_dataset, name='training_iterator')
        self.test_init_op = self.iterator.make_initializer(test_dataset, name='testing_iterator')

        embeddings = tf.nn.embedding_lookup(self.normed_embedding_mat,
                                            features, name='batch_embeddings')
        cell_fw_enc = tf.contrib.rnn.LSTMCell(self.lstm_units, name='lstm_enc_f')
        cell_bw_enc = tf.contrib.rnn.LSTMCell(self.lstm_units, name='lstm_enc_b')
        (output_fw_enc, output_bw_enc), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_enc,
                                                                            cell_bw_enc, embeddings,
                                                                            dtype=tf.float32)

        enc = tf.concat([output_fw_enc, output_bw_enc], axis=-1)
        self.enc_flat = tf.contrib.layers.flatten(enc)

        cell_fw_enc_2 = tf.contrib.rnn.LSTMCell(self.emb_size//2)
        cell_bw_enc_2 = tf.contrib.rnn.LSTMCell(self.emb_size//2)

        (output_fw_enc_2, output_bw_enc_2), _2 = tf.nn.bidirectional_dynamic_rnn(cell_fw_enc_2,
                                                                                 cell_bw_enc_2, enc,
                                                                                 dtype=tf.float32)

        dec = tf.concat([output_fw_enc_2, output_bw_enc_2], axis=-1)
        self.loss = tf.losses.cosine_distance(labels=embeddings,
                                              predictions=dec, axis=1)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
        self.saver = tf.train.Saver()


class PredictionEncoder(Encoder):
    """
    Class for learning sentence encodings using bi–LSTM and maxpooling. Learn by
    optimizing the prediction cross entropy
    """

    def __init__(self, glove_vec_size, top_n_words_to_process,
                 glove_path, labeled_twts, emb_size, lstm_units,
                 num_total_twts, max_seq_length, batch_size):
        """
        Set attributes to be used in subsequent functions and stored output
        """
        Encoder.__init__(self, glove_vec_size, top_n_words_to_process, glove_path, labeled_twts,
                         num_total_twts, max_seq_length, batch_size, lstm_units, emb_size)
        
    def build_lstm_binary_pred_graph(self):
        """
        define data flow and graph for tensorflow model. In this model
        learn the encodings which enable best predictions of sentiment
        """
        tf.reset_default_graph()
        train_data = (self.training_seq, self.training_labels)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)\
                                       .repeat().batch(self.batch_size)
        self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                        train_dataset.output_shapes)
        test_data = (self.testing_seq, self.testing_labels)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            test_data).repeat().batch(self.batch_size)
        features, labels = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(
            train_dataset, name='training_iterator')
        self.test_init_op = self.iterator.make_initializer(
            test_dataset, name='testing_iterator')

        embeddings = tf.nn.embedding_lookup(self.normed_embedding_mat,
                                            features, name='batch_embeddings')
        embeddings_d = tf.nn.dropout(embeddings, 0.5, name='emb_dropout')

        cell_fw_enc = tf.contrib.rnn.LSTMCell(
            self.lstm_units, name='lstm_enc_f')
        cell_bw_enc = tf.contrib.rnn.LSTMCell(
            self.lstm_units, name='lstm_enc_b')

        cell_fw_enc = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw_enc, output_keep_prob=0.5, seed=42)
        cell_bw_enc = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw_enc, output_keep_prob=0.5, seed=42)

        (output_fw_enc, output_bw_enc), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_enc,
                                                                            cell_bw_enc,
                                                                            embeddings_d,
                                                                            dtype=tf.float32)

        cat = tf.concat([output_fw_enc, output_bw_enc], axis=-1)
        maxpool = tf.layers.max_pooling1d(cat, 50, 50)
        self.enc_flat = tf.contrib.layers.flatten(maxpool)
        dropout = tf.nn.dropout(self.enc_flat, 0.5, name='maxpool_dropout')
        preds = tf.layers.dense(self.enc_flat, 2)

        self.loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels, logits=preds)
        #self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.1).minimize(self.loss)
        self.saver = tf.train.Saver()
    
class MaxpoolEncoder(Encoder):
    """
    Build graph for running forward prop on test set, to apply embedding maxpooling.
    Find max value across all words, s.t. dimensionality after maxpooling and 
    flattening is 200
    """

    def __init__(self, glove_vec_size, top_n_words_to_process,
                 glove_path, labeled_twts, emb_size, lstm_units,
                 num_total_twts, max_seq_length, batch_size):
        """
        Set attributes to be used in subsequent functions and stored output
        """
        Encoder.__init__(self, glove_vec_size, top_n_words_to_process, glove_path, labeled_twts,
                         num_total_twts, max_seq_length, batch_size, lstm_units, emb_size)

    def build_maxpool_graph(self, num_epochs=2000):
        """
        Define maxpool of embeddings graph and run forward prop. Maxpool is along
        time dimension, so result is 200–dimensions after pooling and flattening
        """
        tf.reset_default_graph()
        train_data = (self.training_seq, self.training_labels)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            train_data).repeat().batch(self.batch_size)
        #self.iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
        #                                                train_dataset.output_shapes)
        test_data = (self.testing_seq, self.testing_labels)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            test_data).repeat().batch(self.batch_size)
        self.iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                                        test_dataset.output_shapes)
        features, _ = self.iterator.get_next()
        self.train_init_op = self.iterator.make_initializer(
            train_dataset, name='training_iterator')
        self.test_init_op = self.iterator.make_initializer(
            test_dataset, name='testing_iterator')

        embeddings = tf.nn.embedding_lookup(
            self.normed_embedding_mat, features, name='batch_embeddings')

        emb_maxpool = tf.layers.max_pooling1d(embeddings, 50, 50)
        enc_flat = tf.contrib.layers.flatten(emb_maxpool)
        self.enc_flat = enc_flat

        test_set_embeddings = np.zeros(
            [self.batch_size * num_epochs, self.enc_flat.shape[1]])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([self.test_init_op])
            for epoch in range(num_epochs):
                emb_temp = sess.run([self.enc_flat])[0]
                test_set_embeddings[epoch *
                                    self.batch_size:(epoch+1)*self.batch_size] = emb_temp
        test_set_embeddings = test_set_embeddings[0:self.df_test.shape[0], :]
        self.test_set_embeddings = test_set_embeddings
