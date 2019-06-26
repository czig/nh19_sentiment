import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import unicodedata
from sqlalchemy import create_engine
from gensim import corpora
import gensim
#import pyLDAvis.gensim

#import spacy
import spacy
parser = spacy.load('en_core_web_sm')

#add stop words
stop_words = ["Guyana","guyana","year","words","word"]
for word in stop_words:
    parser.vocab[word].is_stop = True

#read SQL database
engine = create_engine('sqlite:///./nh19_fb.db')
posts = pd.read_sql("""select * from posts where created_time > '2019-04-01'""",engine)
comments = pd.read_sql("""select * from comments where created_time > '2019-04-01'""",engine)

posts_list = posts[posts['message'].notnull()].message.to_list()
comments_list = comments[comments['message'].notnull()].message.to_list()

#tokenize and clean posts
def tokenize(messages_list, parser):
    docs_list = []
    for message in messages_list:
        lda_tokens = []
        #convert unicode punctuation to regular ascii punctuation 
        message = message.replace(chr(8216),"'")
        message = message.replace(chr(8217),"'")
        message = message.replace(chr(8218),",")
        message = message.replace(chr(8220),'"')
        message = message.replace(chr(8221),'"')
        message = message.replace(chr(8242),'`')
        message = message.replace(chr(8245),'`')
        #convert remaining unicode characters to closest ascii character
        message = unicodedata.normalize('NFKD',message).encode('ascii','ignore').decode('utf-8')
        #part of speech to include
        allowed_pos = ['NOUN','VERB','PROPN']
        #unicode for 's (right apostrophie followed by s)
        possessive_substr = chr(8217) + 's'
        message_tokens = parser(message)
        #iterate through all tokens in each post
        for token in message_tokens:
            #remove space
            if token.orth_.isspace():
                continue
            #remove punctuation
            elif token.is_punct:
                continue
            #remove urls
            elif token.like_url:
                continue
            #remove emails
            elif token.like_email:
                continue
            #remove stop words
            elif token.is_stop:
                continue
            #remove 's
            elif token.text.find(possessive_substr) > -1:
                continue
            #remove single letters
            elif len(token.text) < 2:
                continue
            #remove unnecessary parts of speech (also removes space and punctuation)
            elif token.pos_ not in allowed_pos:
                continue
            elif token.text in ['lol']:
                continue
            else:
                lda_tokens.append(token.lemma_)
                if token.lemma_ in ['not']:
                    print('Text, Lemma, POS, Tag: ',token.text, token.lemma_, token.pos_, token.tag_)
                    for char in token.lemma_:
                        print(char,ord(char))

        docs_list.append(lda_tokens)

    return docs_list

#tokenize here
docs_list = tokenize(comments_list, parser)

#begin lda
dictionary = corpora.Dictionary(docs_list)
corpus = [dictionary.doc2bow(doc) for doc in docs_list]

#run lda topic modelling
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word = dictionary, passes=15)

topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)

#visualize topics
#lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(lda_display)
