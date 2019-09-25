#!/usr/bin/env python
# coding: utf-8

# In[1]:


from newsapi import NewsApiClient
import random
from pyknp import KNP
import sys
import emoji
import re
import neologdn

key = "1cc54f894659430f86ac6c1686a63c90"
info_elements =["who", "do", "what", "when", "where", "how", "whose"]


# In[2]:


def get_headlines(_category):
    newsapi = NewsApiClient(api_key=key)
    headlines = newsapi.get_top_headlines(category= _category, country='jp')
    if( headlines['totalResults'] > 0 ):
        pass
        #print(headlines["totalResults"])
    else:
        print("条件に合致したトップニュースはありません。")
        
    return headlines


# category = {business、entertainment、general、health、science、sports、technology}

# In[3]:


def get_ramdom_news(headlines):
    news = []
    for headline in headlines['articles']:
        news.append((headline["title"], headline["description"]))

    kiji = random.choice(news)
    
   # o_kiji.append(kiji) #デバッグ用

    midashi = clean_sentence(kiji[0].split("-")[0])


    #utterance = "最近のニュースでは「" + midashi + "」だそうです。\n"

    #print(utterance)
    
    try:
        sentences = []
        lines = kiji[1].split("。")
        for line in lines:
            sentences.append(clean_sentence(line))
    except:
        sentences = kiji[1]
    
    return midashi,sentences


# In[4]:


def select_normalization_representative_notation(fstring):
    """ 正規化代表表記を抽出します
    """
    begin = fstring.find('正規化代表表記:')
    end = fstring.find(">", begin+1)
    daihyous = fstring[begin + len('正規化代表表記:') : end].split("/")
    sentence = daihyous[0]
    for daihyou in daihyous[1:]:
        if(daihyou.find("+") != -1):
            sentence += daihyou.split("+")[1]

    return sentence


# In[5]:


def select_dependency_structure(line):
    """係り受け構造を抽出します
    """

    # KNP
    knp = KNP(option = '-tab -anaphora')

    # 解析
    result = knp.parse(line)

    # 文節リスト
    bnst_list = result.bnst_list()

    # 文節リストをidによるディクショナリ化する
    bnst_dic = dict((x.bnst_id, x) for x in bnst_list)

    tuples = []
    for bnst in bnst_list:
        if bnst.parent_id != -1:
            # (from, to)
            print("bnst_id:{} parent_id:{}\n".format(bnst.bnst_id, bnst.parent_id))
            tuples.append((select_normalization_representative_notation(bnst.fstring), select_normalization_representative_notation(bnst_dic[bnst.parent_id].fstring)))

    return tuples


# In[6]:


def get_gimonshi(bnst, bnst_dic):
    
    fstring = bnst.fstring
    #ノ格
    if fstring.find("ノ格") != -1:
        if (fstring.find("組織名") != -1 or fstring.find("組織名疑") != -1 or fstring.find("地名") != -1 or fstring.find("地名疑") != -1): 
            return "where"
                
        elif (fstring.find("人") != -1 or fstring.find("人名") != -1):
            if bnst_dic[bnst.parent_id].fstring.find("用言:動"):
                return "who"
            else:
                return "whose"
            
        elif(fstring.find("時間") != -1 and fstring.find("SM-主体") == -1):
            return "when"
        
    #ガ格
    elif fstring.find("ガ格") != -1:
        if (fstring.find("人") != -1 or fstring.find("人名") != -1):
            return "who"
        else:
            return "what"
        
        
    #ヲ格
    elif fstring.find("ヲ格") != -1:
        return "what"
            
    #二格
    elif fstring.find("ニ格") != -1:
        if (fstring.find("時間") != -1 and fstring.find("SM-主体") == -1):
            return "when"
        elif (fstring.find("組織名") != -1 or fstring.find("組織名疑") != -1 or fstring.find("地名") != -1 or fstring.find("地名疑") != -1): 
            return "where"
        elif (fstring.find("人") != -1 or fstring.find("人名") != -1):
            return "who"
        else:
            return "what"
        
    #へ格
    elif fstring.find("ヘ格") != -1:
        if (fstring.find("組織名") != -1 or fstring.find("組織名疑") != -1 or fstring.find("地名") != -1 or fstring.find("地名疑") != -1): 
            return "where"
        else:
            return "what"
        
    #ト格
    elif fstring.find("ト格") != -1:
        if (fstring.find("人") != -1 or fstring.find("人名") != -1):
            return "who"
        else:
            return "what"
                
    #デ格
    elif fstring.find("デ格") != -1:
        if (fstring.find("組織名") != -1 or fstring.find("組織名疑") != -1 or fstring.find("地名") != -1 or fstring.find("地名疑") != -1): 
            return "where"
        elif bnst_dic[bnst.parent_id].fstring.find("用言:動") != -1:
                return "how"
                
        
     #ハ
    elif fstring.find("<ハ>") != -1:
        if (fstring.find("組織名") != -1 or fstring.find("組織名疑") != -1 or fstring.find("地名") != -1 or fstring.find("地名疑") != -1): 
            return "where"
        else:
            return "who"
        
    #述語
    elif fstring.find("用言:動") != -1:
            return "do"
        
    #無格
    elif fstring.find("無格") != -1:
        if(fstring.find("時間") != -1 and fstring.find("SM-主体") == -1):
            return "when"
        elif(fstring.find("組織名") != -1 or fstring.find("組織名疑") != -1  or fstring.find("地名") != -1 or fstring.find("地名疑") != -1): 
            return "where"
        
    #形容詞
    elif fstring.find("用言:形") != -1:
        return "how"
    
    else:
        return None


# In[7]:


def generate_knowledge(sentence):
    ##knpで解析
    knp = KNP(option = '-tab -anaphora')
    result = knp.parse(sentence.replace(" ", ""))
    bnst_list = result.bnst_list()

    #文節辞書
    bnst_dic = dict((x.bnst_id, x) for x in bnst_list)

    infos = []
    info = dict((x, None) for x in info_elements)
    for bnst in bnst_list:
        place = get_gimonshi(bnst, bnst_dic)
        
        if(place == None):
            pass
        
        elif info[place] == None:
            info[place] = select_normalization_representative_notation(bnst.fstring)
            
        else:
            infos.append(info)
            del info
            info = dict((x, None) for x in info_elements)
            info[place] = select_normalization_representative_notation(bnst.fstring)
    
    return infos


# In[8]:


def create_infos(topic):
    all_infos = []
    midashi, sentences = get_ramdom_news(get_headlines(topic))
    
    #print(sentences)
    
    for i, line in enumerate(sentences):
        #print(line + "\n")
        infos = generate_knowledge(line)
        for info in infos:
            all_infos.append(info)
    
    return midashi,all_infos,sentences


# In[9]:


def clean_sentence(sentence):
    sentence = re.sub("[(].*?[)]", "", sentence)
    sentence = re.sub("[（].*?[）]", "", sentence)
    sentence = re.sub(re.compile("[!-/:-@[-`{-~]"), '', sentence)
    sentence_without_emoji = "".join(["" if c in emoji.UNICODE_EMOJI else c for c in sentence])
    tmp = re.sub(r'(\d)([,.])(\d+)', r'\1\3', sentence_without_emoji)
    return tmp


# In[14]:


def confirm_parent_bnst(sentence):
    line = sentence.replace(" ","").replace("　","")
    tuples = select_dependency_structure(line)
    for t in tuples:
        print(t[0] + ' => ' + t[1])


# In[ ]:




