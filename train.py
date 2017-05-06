import tensorflow as tf
import numpy as np
import re
from model import seq2seq_model
# word embedding

file = open('vector.txt', 'r')
file.readline()
voc = []
embed = []
k = 0
while file:
    line = file.readline().decode('utf-8')
    k += 1
    if k % 1000 == 0:
        print k
    split = re.split(' ', line)
    voc.append(split[0])
    vec = split[1:101]
    for i in range(len(vec)):
        vec[i] = eval(vec[i])
    embed.append(vec)
    if k > 400000:
        break

model = seq2seq_model(voc, embed, 100, 4)
batch_size = 20
maxi_length = 40

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
"""
try:
    saver = tf.train.Saver()
    saver.restore(sess, "model.ckpt")
    print 'load finished'
except:
    sess.run(tf.global_variables_initializer())
"""
try:
    for epoch in range(10):
        file_post = open('weibo_pair_train_Q.post', 'r')
        file_resp = open('weibo_pair_train_Q.response', 'r')
        feed_dict = {'post':[], 'post_len':[], 'resp':[], 'resp_len':[]}
        while file_post:
            for _ in range(batch_size):
                str_post = file_post.readline().decode('utf-8')
                str_resp = file_resp.readline().decode('utf-8')
                post = re.split(' ', str_post)
                resp = re.split(' ', str_resp)
                post[-1] = post[-1][:-1]
                resp[-1] = resp[-1][:-1]
                post = post + ['</s>']
                resp = resp + ['</s>']
                feed_dict['post'].append(post + [' '] * (maxi_length - len(post)))
                feed_dict['resp'].append(resp + [' '] * (maxi_length - len(resp)))
                feed_dict['post_len'].append(len(post))
                feed_dict['resp_len'].append(len(resp))
            model.step(sess, feed_dict)
except KeyboardInterrupt:
    model.save(sess)