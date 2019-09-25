#!/usr/bin/env python
# coding: utf-8

# In[1]:


from newsapi import NewsApiClient
import random
from pyknp import KNP
from pyknp import Jumanpp
import sys
from News import create_infos

gimonshi = {"だれ":"who", "する":"do", "何":"what", "いつ":"when", "どこ":"where","どのように":"how", "だれの":"whose"}


# In[2]:


def question(bnst, bnst_dic):
    
    fstring = bnst.fstring 
    #ノ格
    if fstring.find("ノ格") != -1:
        if fstring.find("どこ") != -1:
            return "where"
        elif fstring.find("だれ") != -1:
            return "who"
        elif fstring.find("いつ") != -1:
            return "when"
    
    #ガ格
    elif fstring.find("ガ格") != -1:
        if fstring.find("だれ") != -1:
            return "who"
        elif fstring.find("なに") != -1:
            return "what"
        
    #ヲ格
    elif fstring.find("ヲ格") != -1:
       # if bnst_dic[bnst.parent_id].fstring.find("用言:動") != -1:
            #return "do"
       #else:
        return "what"
    
    #二格
    elif fstring.find("ニ格") != -1:      
        if fstring.find("いつ") != -1:
            return "when"
        elif fstring.find("どこ") != -1:
            return "where"
        elif fstring.find("だれ") != -1:
            return "who"
        elif fstring.find("なに") != -1:
            return "what"
        
    #へ格
    elif fstring.find("ヘ格") != -1:
        if fstring.find("どこ") != -1:
            return "where"  
        elif fstring.find("なに") != -1:
            return "what"
        
    #ト格
    elif fstring.find("ト格") != -1:
        if fstring.find("だれ") != -1:
            return "who"
        elif fstring.find("なに") != -1:
            return "what"
        elif fstring.find("どこ") != -1:
            return "where"
        
    #デ格
    elif fstring.find("デ格") != -1:
        if fstring.find("どこ") != -1:
            return "where"   
    
    #無格
    elif fstring.find("無格") != -1:
        if fstring.find("いつ") != -1:
            return "when"
        elif fstring.find("どこ") != -1:
            return "where"   
        elif fstring.find("なに") != -1:
            return "what"
        elif fstring.find("だれ") != -1:
            return "who"
    


# In[3]:


def get_u_gimonshi(sentence):
    line = sentence.replace(" ", "")
    knp = KNP(option = '-tab -anaphora')
    result = knp.parse(line)
    bnst_list = result.bnst_list()
    bnst_dic = dict((x.bnst_id, x) for x in bnst_list)
    
    u_gimonshi = ""
    for bnst in bnst_list:
        place = question(bnst, bnst_dic)
        if place != None:
            u_gimonshi = place

    #print(u_gimonshi)
    return u_gimonshi


# In[25]:


def generate_utterance(u_sen, all_infos):
    #何を聞かれているかを判断
    question = get_u_gimonshi(u_sen)
    
    #質問の答えがどこにあるかを検索
    knp = KNP(option = '-tab -anaphora')
    result = knp.parse(u_sen.replace(" ", ""))
    bnst_list = result.bnst_list()
    search_words = []
    for bnst in bnst_list:
        search_words.append(create_infos.select_normalization_representative_notation(bnst.fstring))
    
    search_point = -1
    for search_word in search_words:
        for i, info in enumerate(all_infos):
            if (search_word in info.values()):
                search_point = i
    
    
    answer = ""
    if search_point == -1:
        answer = "No information"
    
    else:
        if all_infos[search_point][question] != None: #質問の答えがその場所にあるとき
            answer = all_infos[search_point][question] + "です"
        else: #質問の答えがその場所にないとき上下の情報を探索
            if search_point == 0:
                if all_infos[search_point + 1][question] != None: 
                    answer = all_infos[search_point + 1][question] + "です"
                else:
                    answer = "No information"
            else:
                if all_infos[search_point - 1][question] != None:
                    answer = all_infos[search_point - 1][question] + "です"
                elif all_infos[search_point + 1][question] != None:
                        answer = all_infos[search_point + 1][question] + "です"
                else:
                    answer = "No information"
    
    return answer


# In[26]:


def news_uttetance(u_sen, all_infos):
    answer = generate_utterance(u_sen,all_infos)
    return answer
    


# In[27]:


def get_news(topic):
    midashi, all_infos, sentences = create_infos.create_infos(topic)
    return midashi, all_infos, sentences


# In[ ]:




