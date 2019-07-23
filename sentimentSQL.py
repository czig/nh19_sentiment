import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt

#initialize sentiment analysis
sid = SentimentIntensityAnalyzer()

#initialize input sql database
engine = create_engine('sqlite:///./nh19_fb.db')

#initialize output sql databases
raw_engine = create_engine('sqlite:///./raw_sentiment.db')
nh19_engine = create_engine('sqlite:///./nh19_sentiment.db')
guy_engine = create_engine('sqlite:///./guy_sentiment.db')

#TODO: separate pull from posts too
#pull data using sql query
df = pd.read_sql("""select comm.*,
                         post.message as post_message 
                  from comments as comm
                  left join posts as post
                      on comm.parent_id = post.id
                  where comm.message != "" """, engine)
post_df = pd.read_sql("""select * from posts where message != "" """, engine)

#create sentiment scores for comments
df['neg'] = 0.0
df['neu'] = 0.0
df['pos'] = 0.0
df['compound'] = 0.0
for index, row in df.iterrows():
	sentiment = {}
	sentiment['analysis'] = sid.polarity_scores(row['message'])
	df.at[index,'pos'] = sentiment['analysis']['pos']
	df.at[index,'neg'] = sentiment['analysis']['neg']
	df.at[index,'neu'] = sentiment['analysis']['neu']
	df.at[index,'compound'] = sentiment['analysis']['compound']

#output raw sentiment to database
df.to_sql('comments',con=raw_engine,if_exists="replace")

#create sentiment scores for posts 
post_df['neg'] = 0.0
post_df['neu'] = 0.0
post_df['pos'] = 0.0
post_df['compound'] = 0.0
for index, row in post_df.iterrows():
	sentiment = {}
	sentiment['analysis'] = sid.polarity_scores(row['message'])
	post_df.at[index,'pos'] = sentiment['analysis']['pos']
	post_df.at[index,'neg'] = sentiment['analysis']['neg']
	post_df.at[index,'neu'] = sentiment['analysis']['neu']
	post_df.at[index,'compound'] = sentiment['analysis']['compound']

#output raw sentiment to database
post_df.to_sql('posts',con=raw_engine,if_exists="replace")

#make different datasets for each page (comments)
embassy_df = df[df['page'] == 'USEmbassyGeorgetown']
southcom_df = df[df['page'] == 'southcom']
nh_df = df[df['page'] == 'AFSOUTHNewHorizons']
afsouth_df = df[df['page'] == 'AFSouthern']
guyana_df =df[(df['page'] != 'USEmbassyGeorgetown')& (df['page'] != 'southcom') & (df['page'] != 'AFSOUTHNewHorizons') & (df['page'] != 'AFSouthern')]

#make different datasets for each page (posts)
embassy_post_df = post_df[post_df['page'] == 'USEmbassyGeorgetown']
southcom_post_df = post_df[post_df['page'] == 'southcom']
nh_post_df = post_df[post_df['page'] == 'AFSOUTHNewHorizons']
afsouth_post_df = post_df[post_df['page'] == 'AFSouthern']
guyana_post_df =post_df[(post_df['page'] != 'USEmbassyGeorgetown')& (post_df['page'] != 'southcom') & (post_df['page'] != 'AFSOUTHNewHorizons') & (post_df['page'] != 'AFSouthern')]

#output guyana dataframe to database
guyana_df.to_sql('comments',con=guy_engine,if_exists="replace")

#output guyana dataframe to database
guyana_post_df.to_sql('posts',con=guy_engine,if_exists="replace")

#filter embassy and southcom comments for new horizons (comments)
embassyNh_df = embassy_df[(embassy_df['post_message'].str.contains("NH19|New Horizons|NewHorizons|Military",case=False,na=False)) | (embassy_df['message'].str.contains("NH19|New Horizons|NewHorizons|Military|USA|United States",case=False,na=False))]
southcomNh_df = southcom_df[(southcom_df['post_message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False)) | (southcom_df['message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False))]
afsouthNh_df = afsouth_df[(afsouth_df['post_message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False)) | (afsouth_df['message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False))]

filtered_df = pd.concat([embassyNh_df, southcomNh_df, nh_df, afsouthNh_df])

#filter embassy and southcom comments for new horizons (posts)
embassyNh_post_df = embassy_post_df[embassy_post_df['message'].str.contains("NH19|New Horizons|NewHorizons|Military|USA|United States",case=False,na=False)]
southcomNh_post_df = southcom_post_df[southcom_post_df['message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False)]
afsouthNh_post_df = afsouth_post_df[afsouth_post_df['message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False)]

filtered_post_df = pd.concat([embassyNh_post_df, southcomNh_post_df, nh_post_df, afsouthNh_post_df])

print('*************Comments**************')
print('embassy: ',embassyNh_df.shape)
print('southcom: ',southcomNh_df.shape)
print('newHorizons: ',nh_df.shape)
print('afsouth: ',afsouthNh_df.shape)
print('total US related: ',filtered_df.shape)
print('total Guyana: ',guyana_df.shape)
print('all raw records: ',df.shape)
print('*************Posts**************')
print('embassy: ',embassyNh_post_df.shape)
print('southcom: ',southcomNh_post_df.shape)
print('newHorizons: ',nh_post_df.shape)
print('afsouth: ',afsouthNh_post_df.shape)
print('total US related: ',filtered_post_df.shape)
print('total Guyana: ',guyana_post_df.shape)
print('all raw records: ',post_df.shape)

#output NH and guyana dataframes to database
filtered_df.to_sql('comments',con=nh19_engine,if_exists="replace")
filtered_post_df.to_sql('posts',con=nh19_engine,if_exists="replace")
