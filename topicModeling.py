import os.path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import unicodedata
from sqlalchemy import create_engine
from gensim import corpora
import gensim
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
import pickle
import sys

#set max recursion depth
sys.setrecursionlimit(10000)

#custom file import
from dbLogger import *

#do logging as dictated by gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

#import argument parser
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--num_topics", help="Number of topics", type=int, default=10)
arg_parser.add_argument("--num_passes", help="Number of passes", type=int, default=10)
arg_parser.add_argument("--iterations", help="Number of iterations", type=int, default=50)
arg_parser.add_argument("--chunk_size", help="Chunk Size", type=int, default=2000)
arg_parser.add_argument("--update_every", help="Update every flag", type=int, default=0)
arg_parser.add_argument("--date", help="Earliest date for posts/comments in format YYYY-MM-DD", default="2019-04-01")
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
arg_parser.add_argument("--logs", help="If supplied, only do logs and don't generate visualization", action="store_true")
arg_parser.add_argument("--name", help="Test name for tracking", type=str, default='sandbox')
args = arg_parser.parse_args()

dictionary_name = "./tmp/dict_fb_onlyGuy_{0}_{1}.dict".format(args.type,args.date)
corpus_name = "./tmp/corpus_fb_onlyGuy_{0}_{1}.mm".format(args.type,args.date)
docs_name = "./tmp/doc_fb_onlyGuy_{0}_{1}.pkl".format(args.type,args.date)

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)
print("Using start_date of: ",args.date)

#import spacy
import spacy
parser = spacy.load('en_core_web_sm')

#add stop words
stop_words = ["Guyana","guyana","year","words","word"]
for word in stop_words:
    parser.vocab[word].is_stop = True

#read SQL database
engine = create_engine('sqlite:///./nh19_fb.db')
posts = pd.read_sql("""select * from posts 
                       where created_time > '{0}' and 
                       page not in ('AFSOUTHNewHorizons','southcom','AFSouthern')""".format(args.date), engine)
comments = pd.read_sql("""select * from comments 
                           where created_time > '{0}' and 
                           page not in ('AFSOUTHNewHorizons','southcom','AFSouthern')""".format(args.date), engine)

posts_list = posts[posts['message'].notnull()].message.to_list()
comments_list = comments[comments['message'].notnull()].message.to_list()
print('Number of posts:',len(posts_list))
print('Number of comments:',len(comments_list))


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
            elif token.text in ['lol','READ','MORE','NEWS']:
                continue
            elif token.lemma_ in ['say','man']:
                continue
            else:
                lda_tokens.append(token.lemma_.lower())

        docs_list.append(lda_tokens)

    return docs_list

#check if dictionary and corpus are already saved
if os.path.exists(dictionary_name) and os.path.exists(corpus_name) and os.path.exists(docs_name):
    #load dictionary and corpus
    dictionary = corpora.Dictionary.load(dictionary_name)
    corpus = corpora.MmCorpus(corpus_name)
    with open(docs_name,'rb') as docs:
        docs_list = pickle.load(docs)
else:
    #build dicionary and corpus
    #tokenize here
    if args.type == 'comments':
        docs_list = tokenize(comments_list, parser)
    else:
        docs_list = tokenize(posts_list, parser)

    #create dictionary and bag of words (corpus) for lda
    dictionary = corpora.Dictionary(docs_list)
    corpus = [dictionary.doc2bow(doc) for doc in docs_list]
    #save for later
    dictionary.save(dictionary_name)
    corpora.MmCorpus.serialize(corpus_name, corpus)
    with open(docs_name,'wb') as docs:
        pickle.dump(docs_list, docs)

#set variables (override defaults to enable custom logging in db)
alpha = 'auto'
beta = None
doc_size = len(corpus)
dbLogger_inst = dbLogger()
dbLogger_inst.set_values(args.num_topics, args.iterations, args.num_passes, args.update_every, args.chunk_size, alpha, beta, args.type, args.date, doc_size, args.name)
logger.addHandler(dbLogger_inst)

#set callbacks
convergence_callback = gensim.models.callbacks.ConvergenceMetric(logger='shell')
coherence_callback = gensim.models.callbacks.CoherenceMetric(corpus=corpus, dictionary=dictionary, texts=docs_list, coherence='c_v', logger='shell')
perplexity_callback = gensim.models.callbacks.PerplexityMetric(corpus=corpus, logger='shell')
diff_callback = gensim.models.callbacks.DiffMetric(logger='shell')

#run lda topic modelling
print('***************************************')
print('Running LDA model on {0} from {1}'.format(args.type, args.date))
print('num_topics: {0}, iterations: {1}, update_every: {2}, passes: {3}, chunk_size: {4}, alpha: {5}, beta: {6}'.format(args.num_topics, args.iterations, args.update_every, args.num_passes, args.chunk_size, alpha, beta))
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = args.num_topics, iterations=args.iterations, id2word = dictionary, update_every=args.update_every, passes=args.num_passes, chunksize=args.chunk_size, alpha=alpha, eta=beta, callbacks=[convergence_callback, coherence_callback, perplexity_callback, diff_callback])

if not args.logs:
    topics = ldamodel.print_topics(num_words=14)
    for topic in topics:
        print(topic)

    #visualize topics
    vis = pyLDAvis.gensim.prepare(ldamodel,corpus,dictionary)
    pyLDAvis.save_html(vis, './lda_vis/lda_vis_{0}_{1}.html'.format(args.type,args.date))
