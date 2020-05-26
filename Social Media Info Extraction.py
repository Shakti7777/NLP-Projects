#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


tweets = pd.read_csv("C://Users//Sony//Desktop//tweets.csv", encoding= 'ISO-8859-1')


# In[3]:


tweets.head()


# In[4]:


tweets['text'].head()


# In[14]:


#preprocess data

import string
punctuation= string.punctuation
from nltk.corpus import stopwords
stopwords= stopwords.words('english')

def _clean(text):
    text = text.lower()
    text = "".join(x for x in text if x not in punctuation)
    
    words = text.split()
    words = [w for w in words if w not in stopwords]
    
    text = " ".join(words)
    return(text)

_clean('IT is a sample text!!')


# In[15]:


tweets['cleaned']=tweets['text'].apply(_clean)


# In[16]:


tweets[['text','cleaned']]


# In[17]:


#information minning may include
#keywords,phrases,people mentioned,hashtags,URL's/Emails,Person names,company names,sentiment analysis


# In[18]:


#keyword analysis
from collections import Counter
complete_text = "".join(tweets['cleaned'])
words = complete_text.split()
Counter(words).most_common(50)


# In[20]:


#mentions
raw_text = "".join(tweets['text'])
mentions = [w for w in raw_text.split() if w.startswith("@")]

Counter(mentions).most_common(50)


# In[21]:


#hashtags

raw_text = "".join(tweets['text'])
hashtags = [w for w in raw_text.split() if w.startswith("#")]

Counter(hashtags).most_common(50)


# In[22]:



raw_text = "".join(tweets['text'])
hashtags = [w for w in raw_text.split() if w.startswith("#")]
hashtags = [w for w in hashtags if 'demo' not in w.lower()]

Counter(hashtags).most_common(50)


# In[23]:


#links shared
raw_text = "".join(tweets['text'])
links = [w for w in raw_text.split() if w.startswith("http")]

Counter(links).most_common(50)


# In[25]:


from nltk import ngrams
bigrams = ngrams(complete_text.split(),2)
Counter(bigrams).most_common(50)


# In[26]:


#fins an entity name example person name/company name
import nltk
from nltk import word_tokenize,pos_tag,ne_chunk


for text in tweets['text']:
    entities = ne_chunk(pos_tag(word_tokenize(text)))
    for entity in entities:
        if hasattr(entity, 'label'):
            print(entity)
        


# In[29]:


#sentiment of evry tweet
#postive and negative sentiment
import textblob
from textblob import TextBlob


# In[30]:


TextBlob('I hate you').sentiment


# In[31]:


TextBlob('I love you').sentiment


# In[ ]:


##information Linked ,descriptive statistics,time series analysis(how does some particular things/sentiment change over time),
##link with other variables how linking is between 2 things(correlation)
##recommendation engines
## in machine learning algos
##creaet knowledges graph by aalysing unstructured data of what people search n all

