#!/usr/bin/env python
# coding: utf-8

# In[11]:


from bert import modeling
from bert import tokenization
from bert import extract_features
import tensorflow as tf
import numpy as np
import random
import os 


dialog_dir = "./dbdc/dialog_data/"
sys_utterance = "sys_utterance.txt"
bert_output = "./dbdc/dialog_data/sys_bert_output.jsonl"
input_path = "./seq2seq/tweets/input_text_70000.txt"
input_json_path = "./seq2seq/tweets/input_text_70000.jsonl"

tf.app.flags.DEFINE_string('f', '', 'kernel')

topic = {"経済":"business", "芸能":"entertainment", "スポーツ":"sports", "健康":"health"}



# 参照するレイヤーを指定する
TARGET_LAYER = -2
 
# 参照するトークンを指定する
SENTENCE_EMBEDDING_TOKEN = '[CLS]'


# In[12]:

if not os.path.exists(bert_output):
    extract_features.main(dialog_dir + sys_utterance, bert_output)

if not os.path.exists(input_json_path):
    extract_features.main(input_path, input_json_path)


from News import U_gimonshi as news
from dbdc import Search_utterance as dbdc
from seq2seq import seq2seq_bert as seq2seq



def input_embedding(sentence):
    result = []
    output_jsonl = extract_features.call_bert(sentence)
    for feature in output_jsonl['features']:
        if feature['token'] != SENTENCE_EMBEDDING_TOKEN: continue
        for layer in feature['layers']:
            if layer['index'] != TARGET_LAYER: continue
            result.append(layer['values'])
    
    result_array = np.array(result)
    
    return result_array


# In[ ]:

class Reply:
    def __init__(self):
        self.category = random.choice(list(topic.items()))
        self.midashi, self.all_infos, _ = news.get_news(self.category[1])
        
    def get_reply(self, sentence, count):
        if count == 1:
            return self.category[0] + "ニュースについて面白い記事があります"
        elif count == 2:
            return "最近では「" + self.midashi + "」だそうです"
        else:
            print("DBDC or Seq2seq")
            if news.news_uttetance(sentence, self.all_infos) != "No information":
                print("news")
                return news.news_uttetance(sentence, self.all_infos)
            
            else:
                reply_candidate = []
                print("else")
                emb_sen = input_embedding(sentence)
                rep_dbdc = dbdc.utterance_from_dbdc(emb_sen)
                print("DBDC:{}\n".format(rep_dbdc))
                reply_candidate.append(rep_dbdc)
                rep_seq2seq = seq2seq.utterance_from_seq2seq(emb_sen)
                print("Seq2Seq:{}\n".format(rep_seq2seq))
                reply_candidate.append(rep_seq2seq)
                
                return random.choice(reply_candidate)
            
            

