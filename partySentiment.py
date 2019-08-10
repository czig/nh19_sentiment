import numpy as np
import pandas as pd
import gensim
import sys
import os
import re
from sqlalchemy import create_engine
from gensim.models.phrases import Phrases, Phraser
import spacy
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc = {'figure.figsize':(16,9)})
#can change default below (larger lines and labels)
sns.set_context('talk')

#import argument parser
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use model trained with posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Group of pages to use. Default is guy.", choices=['all','nh','guy'],default = 'guy') 
arg_parser.add_argument("--start_date", help="Start date for documents and trained model in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--end_date", help="End date for documents and trained model in format YYYY-MM-DD. Default is 2019-06-22.", default="2019-06-22")
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
arg_parser.add_argument("--show_comments", help="If included, print all comments to terminal after filtering", action="store_true")
args = arg_parser.parse_args()

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)
print("Using start_date of: ",args.start_date)

#create correct engine given parameters
if args.pages == 'guy':
    engine = create_engine('sqlite:///./guy_sentiment.db')
elif args.pages == 'nh':
    engine = create_engine('sqlite:///./nh19_sentiment.db')
elif args.pages == 'all':
    engine = create_engine('sqlite:///./raw_sentiment.db')

parser = spacy.load('en_core_web_sm')

#select all comments/posts
df = pd.read_sql("""select * from {0} where created_time >= '{1}' and created_time <= '{2}'""".format(args.type, args.start_date, args.end_date), engine)

filtered_df = df[df['message'].str.contains('PNC|Granger|APNU|AFC|President|PPP|Jagdeo|Opposition|BJ|Irfaan Ali|Nagamootoo|Prime Minister',case=False,na=False)].copy()

print('Shape of filtered dataframe: ',filtered_df.shape)
if args.show_comments:
    pd.set_option('display.max_colwidth',-1)
    print(filtered_df['message'])

ppp_words = ['ppp','jagdeo','opposition','bj','irfaan_ali','bharrat_jagdeo','opposition_leader']
pnc_words = ['pnc','granger','apnu','afc','president','nagamootoo','president_granger','prime_minister','david_granger','moses_nagamootoo']
documents_dict = filtered_df[filtered_df['message'].notnull()].message.to_dict()
docs_dict = []
for index in documents_dict:
    tokens_list = []
    tokens = parser(documents_dict[index])
    for token in tokens:
        if token.is_stop:
            continue
        elif token.is_punct:
            continue
        elif token.like_url:
            continue
        elif token.like_email:
            continue
        elif len(token.text) < 2:
            continue
        else:
            tokens_list.append(token.text)
    docs_dict.append({'index': index, 'tokens': tokens_list})

tokens_list = [row['tokens'] for row in docs_dict if type(row['tokens']) == str]
tokens_list_filtered = [tokens for tokens in tokens_list if tokens]
bigram = Phraser(Phrases(tokens_list_filtered, min_count=10))
for idx,item in enumerate(docs_dict):
    for token in bigram[item['tokens']]:
        if '_' in token:
            docs_dict[idx]['tokens'].append(token)

for row in docs_dict:
    message = row['tokens']
    index = row['index']
    ppp_score = 0
    pnc_score = 0
    for token in message:
        if token.lower() in ppp_words:
            ppp_score += 1
        if token.lower() in pnc_words:
            pnc_score += 1

    if ppp_score > pnc_score:
        filtered_df.at[index,'party'] = 'PPP'
    elif ppp_score < pnc_score:
        filtered_df.at[index,'party'] = 'PNC'
    else:
        filtered_df.at[index,'party'] = 'Both'

#average sentiment per time period for each topic (every week)
time_df = filtered_df.copy()
time_df['created_time'] = pd.to_datetime(time_df.created_time)
time_df = time_df.groupby([pd.Grouper(key='created_time', freq='W-MON'),'party']).mean().reset_index()
time_df = time_df.fillna(0)
time_df['created_time'] = time_df['created_time'].dt.strftime("%Y-%m-%d")
print('time dataframe')
print(time_df.head())
unique_partys = time_df['party'].unique()
ax = sns.pointplot(x='created_time',y='compound', hue='party', markers=["."]*len(unique_partys), ci=None, data=time_df, palette=sns.color_palette("muted"))
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
#can choose to not show all tick marks
#ax.set_xticks(ax.get_xticks()[::4])
plt.title('Sentiment towards Political Parties Over Time for {0}'.format(args.type.capitalize()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date",fontsize=18)
plt.ylabel("Average Sentiment Score", fontsize = 18)

#sentiment variance per time period for each topic (every week)
time_std_df = filtered_df.copy()
time_std_df['created_time'] = pd.to_datetime(time_std_df.created_time)
time_std_df = time_std_df.groupby([pd.Grouper(key='created_time', freq='W-MON'),'party']).std().reset_index()
time_std_df = time_std_df.fillna(0)
time_std_df['created_time'] = time_std_df['created_time'].dt.strftime("%Y-%m-%d")
print('standard deviation over time dataframe')
print(time_std_df.head())
plt.figure()
ax = sns.pointplot(x='created_time',y='compound', hue='party', markers=["."]*len(unique_partys), ci=None, data=time_std_df, palette=sns.color_palette("muted"))
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.title('Sentiment Std.Dev. towards Political Parties Over Time for {0}'.format(args.type.capitalize()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date", fontsize = 18)
plt.ylabel("Std. Dev. of Sentiment Score", fontsize = 18)

#count per period for each topic (every week)
time_count_df = filtered_df.copy()
time_count_df['created_time'] = pd.to_datetime(time_count_df.created_time)
time_count_df = time_count_df.groupby([pd.Grouper(key='created_time', freq='W-MON'),'party']).size().reset_index(name='counts')
time_count_df = time_count_df.fillna(0)
time_count_df['created_time'] = time_count_df['created_time'].dt.strftime("%Y-%m-%d")
print('count of topic over time dataframe')
print(time_count_df.head())
plt.figure()
ax = sns.pointplot(x='created_time',y='counts', hue='party', markers=["."]*len(unique_partys), ci=None, data=time_count_df, palette=sns.color_palette("muted")) 
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.title('Number of {0} for Political Parties Over Time'.format(args.type.capitalize()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

#average sentiment per topic
party_df = filtered_df.groupby(['party']).mean()['compound'].sort_values(ascending=False)
print('party dataframe')
print(party_df)
plt.figure(figsize=(12,8))
party_df.plot.bar()
plt.xticks(rotation=40, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.35)
plt.xlabel("Party", fontsize = 20)
plt.ylabel("Average Sentiment", fontsize = 20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Average Sentiment for {0} Per Party, {1} to {2}".format(args.type.capitalize(), args.start_date, args.end_date), fontsize = 24)

plt.show()
