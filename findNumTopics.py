import os.path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import unicodedata
from sqlalchemy import create_engine
from gensim import corpora
from matplotlib import pyplot as plt
import gensim
from gensim.models import Phrases
import warnings
import pickle
import sys
from tokenizer import *

#set max recursion depth
sys.setrecursionlimit(10000)

#do logging as dictated by gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

#import argument parser
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments. Defaults to comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--date", help="Earliest date for posts/comments in format YYYY-MM-DD. Defaults to 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
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

#add stop words
stop_words = ["lol","READ","MORE","NEWS"]
stop_lemmas = ["say", "man", "people","know","time","need","want","go","get","year","word","guyana","like","good","thing","come"]

#parts of speech
allowed_pos = ['NOUN', 'VERB', 'PROPN']

#define and instantiate tokenizer
tokenizer_inst = Tokenizer(stop_words=stop_words, stop_lemmas=stop_lemmas, remove_unicode=True, allowed_pos=allowed_pos, lower_token=True, bigrams=True)

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
        docs_list = tokenizer_inst.tokenize(comments_list)
    else:
        docs_list = tokenizer_inst.tokenize(posts_list)

    #create and filter dictionary and create and bag of words (corpus) for lda
    dictionary = corpora.Dictionary(docs_list)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in docs_list]
    #save for later
    dictionary.save(dictionary_name)
    corpora.MmCorpus.serialize(corpus_name, corpus)
    with open(docs_name,'wb') as docs:
        pickle.dump(docs_list, docs)

#run hdp model 
print('***************************************')
print('Running HDP model on {0} from {1}'.format(args.type, args.date))
hdpmodel = gensim.models.hdpmodel.HdpModel(corpus, id2word = dictionary)

print(hdpmodel.suggested_lda_model())
alpha = np.sort(hdpmodel.hdp_to_lda()[0])
print(alpha[::-1])
plt.plot(alpha[::-1])
plt.ylabel('alpha')
plt.show()

