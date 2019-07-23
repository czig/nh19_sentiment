import numpy as np
import pandas as pd
import gensim
import sys
import os
from gensim.test.utils import datapath
from gensim import corpora
from sqlalchemy import create_engine
from tokenizer import *
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc = {'figure.figsize':(11,4)})

#import argument parser
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use model trained with posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Group of pages to use. Default is guy.", choices=['all','nh','guy']) 
arg_parser.add_argument("--num_topics", help="Number of topics in model. Default is 10.", type=int, default=10)
arg_parser.add_argument("--start_date", help="Start date for trained model in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--end_date", help="End date for trained model in format YYYY-MM-DD. Default is 2019-06-22.", default="2019-06-22")
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
args = arg_parser.parse_args()

#specify path for model to load 
file_path = datapath("ldamodel_{0}_{1}_{2}topics_{3}_to_{4}".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date))

#create name for common dictionary
dictionary_name = "./tmp/dict_fb_{0}_{1}_{2}.dict".format(args.type,args.pages,args.start_date)

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)
print("Using start_date of: ",args.start_date)

#add stop words
stop_words = ["lol","READ","MORE","NEWS"]
if args.type == 'comments':
    #stop lemmas for comments
    stop_lemmas = ["say", "man", "people","know","time","need","want","go","get","year","word","guyana","like","good","thing","come","let","think","look","right","day"]
else:
    #stop lemmas for posts
    stop_lemmas = ["say", "man", "people","know","time","need","want","go","get","year","word","guyana","like","good","thing","come","let","think","look","right","day","national","guyanese"]

#parts of speech
allowed_pos = ['NOUN', 'VERB', 'PROPN']

#define and instantiate tokenizer
tokenizer_inst = Tokenizer(stop_words=stop_words, stop_lemmas=stop_lemmas, remove_unicode=True, allowed_pos=allowed_pos, lower_token=True, bigrams=True)

#check if dictionary and corpus are already saved
if os.path.exists(dictionary_name):
    #load dictionary and corpus
    dictionary = corpora.Dictionary.load(dictionary_name)
else:
    sys.exit('Error: dictionary for {0} for {1} starting at {2} does not exist.'.format(args.type, args.pages, args.start_date))

ldamodel = gensim.models.ldamodel.LdaModel.load(file_path)
all_topics = ldamodel.print_topics(num_topics=-1)
for topic in all_topics:
    print(topic)

#create correct engine given parameters
if args.pages == 'guy':
    engine = create_engine('sqlite:///./guy_sentiment.db')
    out_engine = create_engine('sqlite:///./guy_classified.db')
elif args.pages == 'nh':
    engine = create_engine('sqlite:///./nh19_sentiment.db')
    out_engine = create_engine('sqlite:///./nh19_classified.db')
elif args.pages == 'all':
    engine = create_engine('sqlite:///./raw_sentiment.db')
    out_engine = create_engine('sqlite:///./raw_classified.db')

#select all comments/posts
df = pd.read_sql("""select * from {0} where created_time >= '{1}' limit 1000""".format(args.type, args.start_date), engine)

#log size of df
print('Df shape: ', df.shape)

#read topic meta file to get names of topics
meta_file_path = "./models/ldamodel_meta_{0}_{1}_{2}topics_{3}_to_{4}.txt".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date)
with open(meta_file_path,"r") as meta_file:
    json_topics = json.load(meta_file)
topics = {int(topic): json_topics[topic] for topic in json_topics}

df['topic'] = ""
count = 0
for index,row in df.iterrows():
    #comment table and post table both have row called message (no longer using column
    #'post_message' on comments table)
    message = row['message']
    #tokenize one message (both post and comment messages are filtered in their respective tables
    #when generating sentiment scores -- no need to catch null message now)
    tokens = tokenizer_inst.tokenize([message], return_docs=False)
    #convert message to bag of words
    doc = dictionary.doc2bow(tokens)
    #classify as a topic (topic_dist is a list of tuples, where each tuple stores (topic_id, probability))
    topic_dist = ldamodel[doc]
    #take max over second element of tuple (probability)
    most_likely_topic = max(topic_dist, key=lambda item:item[1])
    most_likely_topic_name = topics[most_likely_topic[0]]['name']
    df.at[index, 'topic'] = most_likely_topic_name

    #log status 
    if index % 1000 == 0:
        print('%d %s classified' % (count*1000,args.type))
        count += 1

#average sentiment per time period for each topic (every week)
time_df = df
time_df['created_time'] = pd.to_datetime(time_df.created_time)
time_df = time_df.groupby([pd.Grouper(key='created_time', freq='W-MON'),'topic']).mean().reset_index()
time_df = time_df.fillna(0)
time_df['created_time'] = time_df['created_time'].dt.strftime("%Y-%m-%d")
print('time dataframe')
print(time_df.head())
unique_topics = time_df['topic'].unique()
sns.pointplot(x='created_time',y='compound', hue='topic', markers=["."]*len(unique_topics), ci=None, data=time_df, palette=sns.color_palette("muted"))
plt.title('Sentiment by Topic Over Time', fontsize=24)
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Average Sentiment Score", fontsize = 14)
plt.savefig("./topics/topic_over_time_{0}_{1}_{2}topics_{3}_to_{4}.png".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date))

#sentiment variance per time period for each topic (every week)
time_std_df = df
time_std_df['created_time'] = pd.to_datetime(time_std_df.created_time)
time_std_df = time_std_df.groupby([pd.Grouper(key='created_time', freq='W-MON'),'topic']).std().reset_index()
time_std_df = time_std_df.fillna(0)
time_std_df['created_time'] = time_std_df['created_time'].dt.strftime("%Y-%m-%d")
print('standard deviation over time dataframe')
print(time_std_df.head())
plt.figure()
unique_topics = time_std_df['topic'].unique()
sns.pointplot(x='created_time',y='compound', hue='topic', markers=["."]*len(unique_topics), ci=None, data=time_std_df, palette=sns.color_palette("muted"))
plt.title('Sentiment Std. Dev. by Topic Over Time', fontsize=24)
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Std. Dev. of Sentiment Score", fontsize = 14)
plt.savefig("./topics/topic_std_over_time_{0}_{1}_{2}topics_{3}_to_{4}.png".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date))

#average sentiment per topic
topic_df = df.groupby(['topic']).mean()['compound'].sort_values(ascending=False)
print('topic dataframe')
print(topic_df)
plt.figure()
topic_df.plot.bar()
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Topic", fontsize = 20)
plt.ylabel("Average Sentiment", fontsize = 20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Average Sentiment Per Topic {0} to {1}".format(args.start_date, args.end_date), fontsize = 24)
plt.savefig("./topics/topic_avg_sentiment_{0}_{1}_{2}topics_{3}_to_{4}.png".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date))

#average sentiment per topic
topic_std_df = df.groupby(['topic']).std()['compound'].sort_values(ascending=False)
print('topic sentiment std. dev. dataframe')
print(topic_std_df.head())
plt.figure()
topic_std_df.plot.bar()
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Topic", fontsize = 20)
plt.ylabel("Sentiment Std. Dev.", fontsize = 20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Sentiment Std. Dev. Per Topic {0} to {1}".format(args.start_date, args.end_date), fontsize = 24)
plt.savefig("./topics/topic_std_sentiment_{0}_{1}_{2}topics_{3}_to_{4}.png".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date))

#number of comments per topic
number_df = df.groupby(['topic']).size().sort_values(ascending=False)
print('number dataframe')
print(number_df)
plt.figure()
number_df.plot.bar()
plt.xticks(rotation=400, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Topic", fontsize=20)
plt.ylabel("Total Comments", fontsize=20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Number of comments Per Topic {0} to {1}".format(args.start_date, args.end_date), fontsize = 24)
plt.savefig("./topics/topic_count_{0}_{1}_{2}topics_{3}_to_{4}.png".format(args.type,args.pages,args.num_topics, args.start_date, args.end_date))

#write to database
df.to_sql(args.type,con=out_engine,if_exists="replace",index=False)

plt.show()
