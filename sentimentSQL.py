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

#initialize sql database
engine = create_engine('sqlite:///./nh19_fb.db')
out_engine = create_engine('sqlite:///./nh19_sentiment.db')

#pull data using sql query
df = pd.read_sql("""select comm.*,
                         post.message as post_message 
                  from comments as comm
                  left join posts as post
                      on comm.parent_id = post.id
                  where comm.message != "" """, engine)

#make different datasets for each page
embassy_df = df[df['page'] == 'USEmbassyGeorgetown']
southcom_df = df[df['page'] == 'southcom']
nh_df = df[df['page'] == 'AFSOUTHNewHorizons']

#filter embassy and southcom comments for new horizons
embassyNh_df = embassy_df[(embassy_df['post_message'].str.contains("NH19|New Horizons|NewHorizons|Military",case=False,na=False)) | (embassy_df['message'].str.contains("NH19|New Horizons|NewHorizons|Military|USA|United States",case=False,na=False))]
southcomNh_df = southcom_df[(southcom_df['post_message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False)) | (southcom_df['message'].str.contains("NH19|New Horizons|NewHorizons|Guyana",case=False,na=False))]

filtered_df = pd.concat([embassyNh_df,southcomNh_df,nh_df])
filtered_df['neg'] = 0.0
filtered_df['neu'] = 0.0
filtered_df['pos'] = 0.0
filtered_df['compound'] = 0.0

sentiments = []
for index,row in filtered_df.iterrows():
    sentiment = {}
    sentiment['comment'] = row['message']
    sentiment['post'] = row['post_message']
    sentiment['analysis'] = sid.polarity_scores(row['message'])
    filtered_df.at[index,'pos'] = sentiment['analysis']['pos'] 
    filtered_df.at[index,'neg'] = sentiment['analysis']['neg'] 
    filtered_df.at[index,'neu'] = sentiment['analysis']['neu'] 
    filtered_df.at[index,'compound'] = sentiment['analysis']['compound'] 
    sentiments.append(sentiment)


print('embassy',len(embassyNh_df.index))
print('southcom',len(southcomNh_df.index))
print('newHorizons',len(nh_df.index))
print('total',len(filtered_df.index))

page_stats = filtered_df.groupby("page")
print(page_stats.describe())
print(page_stats.mean())

filtered_df.to_sql('CommentSentiment',con=out_engine,if_exists="replace")
filtered_df.to_excel('Output.xlsx')
