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
from gensim.models import Phrases
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
import pickle
import sys
from pprint import pprint
import json

#set max recursion depth
sys.setrecursionlimit(10000)

#custom file import
from tokenizer import *

#do logging as dictated by gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

#import argument parser
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Group of pages to use. Default is guy.", choices=['all','nh','guy'], default='guy') 
arg_parser.add_argument("--num_topics", help="Number of topics. Default is 10.", type=int, default=10)
arg_parser.add_argument("--num_passes", help="Number of passes. Default is 10.", type=int, default=10)
arg_parser.add_argument("--iterations", help="Number of iterations. Default is 50.", type=int, default=50)
arg_parser.add_argument("--chunk_size", help="Chunk Size. Default is 2000.", type=int, default=2000)
arg_parser.add_argument("--update_every", help="Update every flag. Default is 0 (batch learning).", type=int, default=0)
arg_parser.add_argument("--start_date", help="Earliest date for posts/comments in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--end_date", help="Latest date for posts/comments in format YYYY-MM-DD. Default is 2019-06-22.", default="2019-06-22")
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
arg_parser.add_argument("--logs", help="If supplied, only do logs and don't generate visualization", action="store_true")
arg_parser.add_argument("--name", help="Test name for tracking. Default is 'sandbox'.", type=str, default='sandbox')
args = arg_parser.parse_args()

dictionary_name = "./tmp/dict_fb_{0}_{1}_{2}_to_{3}.dict".format(args.type,args.pages,args.start_date,args.end_date)
corpus_name = "./tmp/corpus_fb_{0}_{1}_{2}_to_{3}.mm".format(args.type,args.pages,args.start_date,args.end_date)
docs_name = "./tmp/doc_fb_{0}_{1}_{2}_to_{3}.pkl".format(args.type,args.pages,args.start_date,args.end_date)

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)
print("Using start_date of: ",args.start_date)
print("Using end_date of: ",args.end_date)

#pages used for facebook pull (USEmbassy included in all intentionally)
if args.pages == 'all':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','dpiguyana','AFSouthern','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']
elif args.pages == 'nh':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','AFSouthern']
elif args.pages == 'guy':
    page_ids = ['USEmbassyGeorgetown','dpiguyana','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy'] 

#add stop words
stop_words = ["lol","READ","MORE","NEWS"]
if args.type == 'comments':
    #stop lemmas for comments
    stop_lemmas = ["say", "man", "people","know","time","need","want","go","get","year","word","guyana","like","good","thing","come","let","think","look","right","day","guyanese","country","sad","ppl","way","yuh","be","guy","comment","lol"]
else:
    #stop lemmas for posts
    stop_lemmas = ["say", "man", "people","know","time","need","want","go","get","year","word","guyana","like","good","thing","come","let","think","look","right","day","national","guyanese","news"]

#parts of speech
allowed_pos = ['NOUN', 'VERB', 'PROPN']

#define and instantiate tokenizer
remove_unicode = True
lower_token=True
bigrams=True
tokenizer_inst = Tokenizer(stop_words=stop_words, stop_lemmas=stop_lemmas, remove_unicode=remove_unicode, allowed_pos=allowed_pos, lower_token=lower_token, bigrams=bigrams)

#read SQL database 
engine = create_engine('sqlite:///./nh19_fb.db')
all_documents = pd.read_sql("""select * from {0} where created_time >= '{1}' and created_time <= '{2}'""".format(args.type,args.start_date,args.end_date), engine)

#filter for desired pages
relevant_documents = all_documents[all_documents['page'].isin(page_ids)]

#convert to list for tokenizing
documents_list = relevant_documents[relevant_documents['message'].notnull()].message.to_list()
print('Number of {0}:'.format(args.type),len(documents_list))

#check if dictionary and corpus are already saved, and if so, load then and do not tokenize 
if os.path.exists(dictionary_name) and os.path.exists(corpus_name) and os.path.exists(docs_name):
    #load dictionary and corpus
    dictionary = corpora.Dictionary.load(dictionary_name)
    corpus = corpora.MmCorpus(corpus_name)
    with open(docs_name,'rb') as docs:
        docs_list = pickle.load(docs)
else:
    #build dicionary and corpus
    #tokenize here
    docs_list = tokenizer_inst.tokenize(documents_list)

    #create and filter dictionary and create and bag of words (corpus) for lda
    dictionary = corpora.Dictionary(docs_list)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in docs_list]
    #save for later
    dictionary.save(dictionary_name)
    corpora.MmCorpus.serialize(corpus_name, corpus)
    with open(docs_name,'wb') as docs:
        pickle.dump(docs_list, docs)
    #save arguments used for creating dictionary and corpus (dictionary and corpus metadata)
    tokenize_meta_path = "./tmp/meta_fb_{0}_{1}_{2}_to_{3}.txt".format(args.type,args.pages,args.start_date,args.end_date)
    tokens_meta_dict = {
        'stop_words': stop_words,
        'stop_lemmas': stop_lemmas,
        'allowed_pos': allowed_pos,
        'remove_unicode': remove_unicode,
        'lower_token': lower_token,
        'bigrams': bigrams
    }
    with open(tokenize_meta_path,"w") as tokens_meta_file:
        tokens_meta_file.write(json.dumps(tokens_meta_dict))

#log number of unique tokens and number of documents
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

#set variables (keep values for saving model meta file)
alpha = 'auto'
beta = 'auto' 
doc_size = len(corpus)

#set callbacks
convergence_callback = gensim.models.callbacks.ConvergenceMetric(logger='shell')
coherence_callback = gensim.models.callbacks.CoherenceMetric(corpus=corpus, dictionary=dictionary, texts=docs_list, coherence='u_mass', logger='shell')
perplexity_callback = gensim.models.callbacks.PerplexityMetric(corpus=corpus, logger='shell')
diff_callback = gensim.models.callbacks.DiffMetric(logger='shell')

#run lda topic modelling
print('***************************************')
print('Running LDA model on {0} from {1} to {2}'.format(args.type, args.start_date, args.end_date))
print('num_topics: {0}, iterations: {1}, update_every: {2}, passes: {3}, chunk_size: {4}, alpha: {5}, beta: {6}'.format(args.num_topics, args.iterations, args.update_every, args.num_passes, args.chunk_size, alpha, beta))
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = args.num_topics, iterations=args.iterations, id2word = dictionary, update_every=args.update_every, eval_every=1, passes=args.num_passes, chunksize=args.chunk_size, alpha=alpha, eta=beta, callbacks=[convergence_callback, coherence_callback, perplexity_callback, diff_callback])

#save model
most_recent_date = relevant_documents.created_time.max()[:10]
file_path = datapath("ldamodel_{0}_{1}_{2}topics_{3}_to_{4}".format(args.type,args.pages,args.num_topics,args.start_date,args.end_date))
ldamodel.save(file_path)

#save meta information for run to track what arguments were used 
meta_file_path = "./models/ldamodel_meta_{0}_{1}_{2}topics_{3}_to_{4}.txt".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date)
meta_dict = {
    'dictionary': dictionary_name,
    'corpus': corpus_name,
    'docs': docs_name,
    'num_topics': args.num_topics,
    'iterations': args.iterations,
    'update_every': args.update_every,
    'num_passes': args.num_passes,
    'chunksize': args.chunk_size,
    'alpha': alpha,
    'beta': beta
}
with open(meta_file_path,"w") as meta_file:
    meta_file.write(json.dumps(meta_dict))

#print topics in order of significance
sig_topics = ldamodel.print_topics(num_topics=-1)
for topic in sig_topics:
    print(topic)

#write topics to text file for classifying comments later
topics_file_path = "./models/ldamodel_topics_{0}_{1}_{2}topics_{3}_to_{4}.txt".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date)
json_topics = {str(topic[0]): {"name": "", "words": topic[1]} for topic in sig_topics}
with open(topics_file_path,"w") as topic_file:
    topic_file.write(json.dumps(json_topics))

#create visualizations 
topics = ldamodel.top_topics(corpus=corpus, dictionary=dictionary, texts=docs_list, coherence='u_mass')
avg_topic_coherence = sum([t[1] for t in topics]) / args.num_topics
print('average topic coherence: %.4f' % avg_topic_coherence)
#print topics to terminal 
pprint(topics)
#build plot for each topic coherence
topic_coherences = [topic[1] for topic in topics]
plt.figure(0)
plt.plot(topic_coherences)
plt.ylabel('Coherence')
#build plot for first 10 topics topic (show word distribution)
for i,topic in enumerate(topics):
    #topic is a tuple, with first element list of tuples and second element topic coherence
    topic_coherence = topic[1]
    words = [element[1] for element in topic[0]]
    coherences = [element[0] for element in topic[0]]
    if i < 10:
        plt.figure(i+1)
        plt.bar(words, coherences)
        plt.xticks(rotation=40, ha='right')
        plt.subplots_adjust(bottom=0.3)
        plt.ylabel("Probability")
        plt.title('Topic #{0}, coherence: {1}'.format(i, round(topic_coherence,3)), fontsize=24)

#visualize topics
vis = pyLDAvis.gensim.prepare(ldamodel,corpus,dictionary)
pyLDAvis.save_html(vis, './lda_vis/lda_vis_{0}_{1}_{2}topics_{3}_to_{4}.html'.format(args.type,args.pages,args.num_topics,args.start_date,args.end_date))

#show plots after saving pyLDAvis
plt.show()
