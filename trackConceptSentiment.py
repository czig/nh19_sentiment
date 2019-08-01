import numpy as np
import pandas as pd
import gensim
import sys
import os
from sqlalchemy import create_engine
from tokenizer import *
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
arg_parser.add_argument("--name", help="Name of concept to track sentiment for", type=str, required=True)
arg_parser.add_argument("--words", help="Words that constitute the concept (separate each word with a space, no hard brackets, no quotes)", type=str, nargs="+", required=True)
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
args = arg_parser.parse_args()

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)
print("Using start_date of: ",args.start_date)

print("Concept: %s" % args.name)
print('Using words: ', args.words)

#create correct engine given parameters
if args.pages == 'guy':
    engine = create_engine('sqlite:///./guy_sentiment.db')
elif args.pages == 'nh':
    engine = create_engine('sqlite:///./nh19_sentiment.db')
elif args.pages == 'all':
    engine = create_engine('sqlite:///./raw_sentiment.db')

#select all comments/posts
df = pd.read_sql("""select * from {0} where created_time >= '{1}' and created_time <= '{2}'""".format(args.type, args.start_date, args.end_date), engine)


filtered_df = df[df['message'].str.contains('|'.join(args.words),case=False,na=False)]

print('Shape of filtered dataframe: ',filtered_df.shape)

#average sentiment per time period for each topic (every week)
time_df = filtered_df
time_df['created_time'] = pd.to_datetime(time_df.created_time)
time_df = time_df.groupby([pd.Grouper(key='created_time', freq='W-MON')]).mean().reset_index()
time_df = time_df.fillna(0)
time_df['created_time'] = time_df['created_time'].dt.strftime("%Y-%m-%d")
print('time dataframe')
print(time_df.head())
ax = sns.pointplot(x='created_time',y='compound', ci=None, data=time_df, color='blue')
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
#can choose to not show all tick marks
#ax.set_xticks(ax.get_xticks()[::4])
plt.title('Sentiment towards {0} Over Time for {1}'.format(args.name,args.type.capitalize()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date",fontsize=18)
plt.ylabel("Average Sentiment Score", fontsize = 18)

#sentiment variance per time period for each topic (every week)
time_std_df = filtered_df
time_std_df['created_time'] = pd.to_datetime(time_std_df.created_time)
time_std_df = time_std_df.groupby([pd.Grouper(key='created_time', freq='W-MON')]).std().reset_index()
time_std_df = time_std_df.fillna(0)
time_std_df['created_time'] = time_std_df['created_time'].dt.strftime("%Y-%m-%d")
print('standard deviation over time dataframe')
print(time_std_df.head())
plt.figure()
ax = sns.pointplot(x='created_time',y='compound', ci=None, data=time_std_df, color='blue')
ax.grid(True)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt.title('Sentiment Std.Dev. towards {0} Over Time for {1}'.format(args.name,args.type.capitalize()), fontsize=24)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.subplots_adjust(bottom=0.2)
plt.xlabel("Created Date", fontsize = 18)
plt.ylabel("Std. Dev. of Sentiment Score", fontsize = 18)

plt.show()
