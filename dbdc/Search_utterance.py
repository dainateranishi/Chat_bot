#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import numpy as np
#from .bert import modeling
#from .bert import tokenization
#from .bert import extract_features
import tensorflow as tf
import csv



courpse_dir = "./DBDC2_ref/IRS"
dialog_dir = "./dbdc/dialog_data/"
user_utterance = "user_utterance.txt"
sys_utterance = "sys_utterance.txt"
bert_output = "./dbdc/dialog_data/sys_bert_output.jsonl"
sys_embedding = "./dbdc/dialog_data/sys_embedding.tsv"

#tf.app.flags.DEFINE_string('f', '', 'kernel')

# 参照するレイヤーを指定する
TARGET_LAYER = -2
 
# 参照するトークンを指定する
SENTENCE_EMBEDDING_TOKEN = '[CLS]'


# In[2]:


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


# In[13]:

    

def load_sys_utterance():
    with open(bert_output, 'r') as f:
        output_jsons = f.readlines()
    embedding_list = []
    for output_json in output_jsons:
        output = json.loads(output_json)
        for feature in output['features']:
            if feature['token'] != SENTENCE_EMBEDDING_TOKEN: continue
            for layer in feature['layers']:
                if layer['index'] != TARGET_LAYER: continue
                embedding_list.append(layer['values'])
                    
    embedding_result =  dict((i, np.array(line)) for i,line in enumerate(embedding_list))
    return embedding_result


# In[4]:


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# In[48]:


def search_utterance(u_utterance, embedding_dict):
    u_embedding = u_utterance
    cos_sim_dict = {}
    for index, value in embedding_dict.items():
        cos_sim_dict[index] = cos_sim(u_embedding, value)
    
    search_point = max(cos_sim_dict, key = cos_sim_dict.get)
    
    with open(dialog_dir + user_utterance) as u:
        lines = u.readlines()
    
    return search_point, lines[search_point]


# In[49]:


sys_emb = load_sys_utterance()


# In[1]:


def utterance_from_dbdc(sentence):
    
    point,sys_sen = search_utterance(sentence, sys_emb)
    
    return sys_sen


# In[ ]:




