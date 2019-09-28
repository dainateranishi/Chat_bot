#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf

tf.enable_eager_execution()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import unicodedata
import re
import numpy as np
import os
import time
from pyknp import Juman
import sys
#from .bert import modeling
#from .bert import tokenization
#from .bert import extract_features
import json
import time
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import random
#tf.app.flags.DEFINE_string('f', '', 'kernel')

print(tf.__version__)
print(tf.keras.__version__)

#extract_features.setting_bert()


# In[2]:


#sp = spm.SentencePieceProcessor()
#spm.SentencePieceTrainer.Train("--input=./tweets/target_text.txt --model_prefix=trained_model--vocab_size=8000")


# In[3]:


START_ID = 1
END_ID = 2
PAD_ID = 16000

input_path = "./seq2seq/tweets/input_text_70000.txt"
output_wakachi = "./seq2seq/tweets/target_wakachi_70000.txt"
out_voc_path = "./seq2seq/tweets/target_text_voc.txt"
#spm_voc = "./trained_model--vocab_size"
#spm_model = "./trained_model--vocab_size"

BATCH_SIZE = 256
bert_dim = 768
embedding_dim = 256
vocab_out_size = 16001
units = 1024

#bucketの種類
_buckets = [(20,20), (40,40), (60,60), (80,80), (100,100), (140, 145)]




print(tf.__version__)
print(tf.keras.__version__)


# ## 22万文から5万文に変更

# In[4]:


#inputのテンソルを付くる
class InputEmbedding():
    def __init__(self, path):
        self.path_txt = path
        self.path_jsonl = path.replace(".txt", ".jsonl")
        
        if os.path.exists(self.path_jsonl):
            pass
        else:
            self.create_jsonl()
        
        #jsonlファイルを作る
    def create_jsonl(self):
        extract_features.main(self.path_txt, self.path_jsonl)
    
    #inputのtensorを作る
    def create_input_tensor(self):
        print("Start Creating input-tensor\n")
        input_tensor = []
        i = 0
        with open(self.path_jsonl, 'r') as f:
            output_jsonls = f.readlines()
            sentence = []
            for i, output_json in enumerate(output_jsonls[:70000]):
                #sys.stdout.write("{}/{}\n".format(i, len(output_jsonls[:70000])))
                output = json.loads(output_json)
                for feature in output['features']:
                    if feature['token'] == "[CLS]": 
                        continue
                    elif feature['token'] == "[SEP]":
                        input_tensor.append(sentence)
                        del sentence
                        sentence = []
                    else:
                        for layer in feature['layers']:
                            sentence.append(layer['values'])
        print("Success Create Input-tensor\n")
                
        return input_tensor


# In[5]:


#outputの作成
class OutputIndex():
    def __init__(self, voc_path):
        self.voc_path = voc_path
        self.word2idx = {}
        self.idx2word = {}
        
        self.create_index()
        
    def create_index(self):
        with open(self.voc_path, mode = "r") as voc:
            lines = voc.readlines()
            self.idx2word = dict((i,sen.rstrip("\n"))for i, sen in enumerate(lines[:16000]))
            self.idx2word[PAD_ID] = "PAD"
            
            for index, word in self.idx2word.items():
                self.word2idx[word] = index
            
    def create_output_tensor(self, path):
        print("Start Creating output-tensor\n")
        output_tensor = []
    
        with open(path, mode = "r") as out:
            lines = out.readlines()
            for i, line in enumerate(lines[:70000]):
                #sys.stdout.write("{}/{}\n".format(i,len(lines[(22400)*(s_line-1):])))
                #sys.stdout.flush()
                
                line = line.rstrip("\n")
                splitted_list = line.split(" ")
                splitted_list.insert(0, self.idx2word["2"])
                splitted_list.insert(len(splitted_list),self.idx2word["3"])
                #sys.stdout.write("{}\n".format(splitted_list))
                output_tensor.append([self.word2idx.get(w, self.word2idx["<UNK>"]) for w in splitted_list])
        print("Success Create target-tensor\n")
        
        return output_tensor


# In[6]:


#trensorの中の最大長を返す
def max_length(tensor):
    return max(len(t) for t in tensor)


# In[7]:


def load_dataset(path_out, input_obj, output_obj):
    input_tensor = input_obj.create_input_tensor()
    output_tensor = output_obj.create_output_tensor(path_out)

    
    data_set_train = [[] for _ in _buckets]
    for i in range(len(input_tensor)):
        for bucket_id, (inp_size, out_size) in enumerate(_buckets):
            if len(input_tensor[i]) < inp_size and len(output_tensor[i]) < out_size:
                data_set_train[bucket_id].append([input_tensor[i], output_tensor[i]])
                break
                
    del input_tensor
    del output_tensor

    bucket_sizes_train = [len(data_set_train[b]) for b in range(len(_buckets))] 
    print(bucket_sizes_train)
    
    inp_ten, out_ten = Padding(data_set_train)
    return inp_ten, out_ten


# In[ ]:


def Padding(data):
    encoder_size, decoder_size = _buckets[0]
    #encoder_start = [[START_ID] * 768]
    encoder_end = [[EOS_ID]* 768]
    encoder_inputs, decoder_inputs = [], []
    for i in range(len(data[0])):
        encoder_input, decoder_input = random.choice(data[0])
        encoder_pad = [[PAD_ID] * 768] * (encoder_size - len(encoder_input)-1) 
        encoder_inputs.append(list(list(reversed(encoder_input)) + encoder_end + encoder_pad)) 

        decoder_pad = [PAD_ID] * (decoder_size - len(decoder_input))                
        decoder_inputs.append(decoder_input + [PAD_ID] * (decoder_size - len(decoder_input))) 
        
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []


    for batch_idx in range(len(data[0])):                                  
        batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for length_idx in range(encoder_size)], dtype=np.float32))


    for batch_idx in range(len(data[0])): 
        batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for length_idx in range(decoder_size)], dtype=np.int32))

    return batch_encoder_inputs, batch_decoder_inputs


# In[9]:


def gru(units):
    #if tf.test.is_gpu_available():
       # print("OK")
       # config = tf.ConfigProto()
       #config.gpu_options.allow_growth = True
        
        #return tf.keras.layers.CuDNNGRU(units,return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    #else:
    return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid',recurrent_initializer='glorot_uniform')


# In[10]:


class Encoder(tf.keras.Model):
    def __init__(self,enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# In[11]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz,self.dec_units))


# In[12]:


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0.)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


# In[13]:


def train(inp_obj, out_obj):
    #datasetの作成
    #list型でinputとoutputを作成
    encoder_inputs, decoder_inputs = load_dataset(output_wakachi,inp_obj, out_obj)
    EPOCHS = 1000
    
    #学習開始
    for epoch in range(EPOCHS):
        start = time.time()
    
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
    
        #テンソル型のデータ作成
        tensor = tf.data.Dataset.from_tensor_slices((encoder_inputs, decoder_inputs)).shuffle(len(encoder_inputs))
        
        N_BATCH = len(encoder_inputs)/BATCH_SIZE
        tensor = tensor.batch(BATCH_SIZE, drop_remainder=True)
        
        for (batch, (inp, out)) in enumerate(tensor):
            loss = 0
            with tf.GradientTape() as tape:
                
                enc_output, enc_hidden = encoder(inp, hidden)
            
                dec_hidden = enc_hidden
            
                dec_input = tf.expand_dims([out_obj.word2idx["<GO>"]] * BATCH_SIZE, 1)       
            
                # Teacher forcing - feeding the target as the next input
                for t in range(1, out.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                   
                    loss += loss_function(out[:, t], predictions)
                
                    # using teacher forcing
                    dec_input = tf.expand_dims(out[:, t], 1)
                
            batch_loss = (loss / int(out.shape[1]))
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables   
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))
        
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 100 == 0:
               checkpoint.save(file_prefix = checkpoint_prefix)
    
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / N_BATCH))
        total_losses.append(total_loss / N_BATCH)
        
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    


# In[14]:


#１文をidに変換
def one_sentence(sentence):
    output_jsonls = extract_features.call_bert(sentence)
    feature = output_jsonls["features"]
    #1文の中の単語id
    sen = []
    for feature in feature:
        if feature['token'] == "[CLS]": 
            continue
        elif feature['token'] == "[SEP]":
            break
        else:
            for layer in feature['layers']:
                sen.append(layer['values'])
    
    #print(sen)
    return sen


# In[15]:


def evaluate(emb_sentence, encoder, decoder, out_obj):
    
    inp_bert_id = emb_sentence
    #インプットがどのバケツに入るかを識別
    #for bucket_id, (inp_size, out_size) in enumerate(_buckets):
        #if len(inp_bert_id) < inp_size:
    inp_bucket_len, out_bucket_len = _buckets[0]
            
    #Padding
    inp_sen = []
    inp_end = [[END_ID]* 768]
    inp_pad = [[PAD_ID] * 768] * (inp_bucket_len - len(inp_bert_id)-1) 
    inp_sen.append(list(list(reversed(inp_bert_id)) + inp_end + inp_pad))
    inp_sen = tf.convert_to_tensor(inp_sen, dtype = tf.float32)
    #print(type(in))
    
    attention_plot = np.zeros((out_bucket_len,inp_bucket_len ))
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inp_sen, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([out_obj.word2idx["<GO>"]], 0)

    for t in range(out_bucket_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += out_obj.idx2word[predicted_id]

        if out_obj.idx2word[predicted_id] == "<EOS>":
            return result, emb_sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, emb_sentence, attention_plot


# In[16]:


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


# In[17]:


def translate(emb_sentence, encoder, decoder, out_obj):
    result, sentence, attention_plot = evaluate(emb_sentence, encoder, decoder, out_obj)
    
    return result
        
    #print('Input: {}'.format(sentence))
   # print('Predicted translation: {}'.format(result))
    
    #juman = Jumanpp()
    #u_analysis = juman.analysis(sentence)
   # u_sen = []
    #for m in u_analysis.mrph_list():
        #u_sen.append(str(m.midasi))
        
   # s_analysis = juman.analysis(result)
   # s_sen = []
    #for n in s_analysis.mrph_list():
        #s_sen.append(str(n.midasi))
   
    
    #attention_plot = attention_plot[:len(s_sen), :len(u_sen)]
    #plot_attention(attention_plot, u_sen, s_sen)


# In[20]:


optimizer = tf.train.AdamOptimizer()
total_losses = []
encoder = Encoder(units, BATCH_SIZE)
decoder = Decoder(vocab_out_size, embedding_dim, units, BATCH_SIZE)
checkpoint_dir = './seq2seq/training_checkpoints_with_bert_3'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")               
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder,decoder=decoder)
inp_obj = InputEmbedding(input_path)
out_obj = OutputIndex(out_voc_path)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print("Setting complite in seq2seq\n")
#train(inp_obj, out_obj)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[1]:


#def setting_seq2seq():
    #optimizer = tf.train.AdamOptimizer()
    #total_losses = []
    #encoder = Encoder(units, BATCH_SIZE)
    #decoder = Decoder(vocab_out_size, embedding_dim, units, BATCH_SIZE)
    #checkpoint_dir = './training_checkpoints_with_bert'
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")               
    #checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder,decoder=decoder)
    #inp_obj = InputEmbedding(input_path)
    #out_obj = OutputIndex(out_voc_path)
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    #return encoder, decoder, inp_obj, out_obj


# In[2]:


def utterance_from_seq2seq(emb_sentence):
    s_utterance = translate(emb_sentence, encoder, decoder, out_obj)
    return s_utterance


# In[3]:


def train():
    train(inp_obj, out_obj)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[ ]:




