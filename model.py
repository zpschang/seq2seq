import tensorflow as tf
import numpy as np
from tensorflow.contrib.lookup import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

def weight(x, y):
    return tf.Variable(tf.truncated_normal([x, y], mean=0, stddev=0.08))
    #return tf.Variable(tf.constant(0.0, shape=[x, y]))

def bias(y):
    #return tf.Variable(tf.truncated_normal([y], mean=0.3, stddev=0.3))
    return tf.Variable(tf.constant(0.1, shape=[y]))

def cross_entropy(x, y):
    return tf.reduce_mean(-tf.reduce_sum(x * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), 1))

class Model:
    def __init__(self, vocab, embed, lstm_size, layer_num):
        # input: name->tensor
        self.learning_rate = 0.5
        self.input = {}
        self.input['post'] = tf.placeholder(tf.string, shape=(None, None)) # batch*len
        self.input['post_len'] = tf.placeholder(tf.int32, shape=(None)) # batch
        self.input['resp'] = tf.placeholder(tf.string, shape=(None, None)) # batch*len
        self.input['resp_len'] = tf.placeholder(tf.int32, shape=(None)) # batch

        # symbol->index
        self.symbols = tf.Variable(vocab, trainable=False, name='symbols') # voc_size
        self.sym2index = HashTable(KeyValueTensorInitializer(self.symbols, 
            tf.Variable(np.array(range(len(vocab)), dtype=tf.int32), trainable=False))
            , -1)
        self.index2sym = HashTable(KeyValueTensorInitializer(tf.Variable(np.array(range(len(vocab)), dtype=tf.int32), trainable=False), self.symbols), -1)
        
        batch_size = tf.shape(self.input['post'])[0]
        max_len = tf.shape(self.input['post'])[1]

        post = self.sym2index.lookup(self.input['post']) # batch*len
        resp = self.sym2index.lookup(self.input['resp']) # batch*len
        resp_shift = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32) * GO_ID,
                            tf.split(resp, [max_len-1, 1], 1)[0]], 1) # batch*len
        std_output = tf.one_hot(resp, len(vocab))

        post_embed = tf.nn.embedding_lookup(embed, post) # batch*len*embed_size
        resp_embed = tf.nn.embedding_lookup(embed, resp_shift) # batch*len*embed_size

        # encoder
        cell_encoder = MultiRNNCell([BasicLSTMCell(lstm_size) for _ in range(layer_num)])
        output, state = tf.nn.dynamic_rnn(cell_encoder, post_embed, sequence_length=self.input['post_len'])
        
        cell_decoder = MultiRNNCell([BasicLSTMCell(lstm_size) for _ in range(layer_num)])
        decoder_output, _ = tf.nn.dynamic_rnn(cell_decoder, resp_embed, sequence_length=self.input['resp_len'], initial_state=state)
        # resp_output: batch*len*lstm_size

        W = weight(lstm_size, len(vocab))
        B = bias(len(vocab))
        poss = tf.nn.softmax(tf.matmul(decoder_output, W) + B) # batch*len*voc_size
        loss = cross_entropy(std_output, poss) # []
        int_output = tf.argmax(poss, axis=2) # batch*len
        is_equal = tf.equal(int_output, resp) # batch*len
        acc = tf.reduce_mean(tf.reduce_mean(tf.cast(is_equal, tf.float32), axis=1)) # []
        str_output = self.index2sym.lookup(int_output) # batch*len

        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.params = tf.trainable_variables()

        self.output = {}
        self.output['loss'] = loss
        self.output['acc'] = acc
        self.output['output'] = str_output
    
    def print_params(self):
        for param in self.params:
            print('%s: %s' % (param.name, param.get_shape()))
    
    def step(self, sess, feed_input):
        out_name = [name for name, value in self.output]
        out_tensor = [value for name, value in self.output]
        print out_name, '=',
        res = sess.run(out_tensor + [self.train], feed_dict={self.input[name]:feed_input[name] for name in self.input})
        print res
    
    def inference(self, sess, feed_input):
        out_name = [name for name, value in self.output]
        out_tensor = [value for name, value in self.output]
        print out_name, '=',
        res = sess.run(out_tensor, feed_dict={self.input[name]:feed_input[name] for name in self.input})
        print res