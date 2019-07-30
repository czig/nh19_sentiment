import pandas as pd
import numpy as np
from os import path
from PIL import Image
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
sns.set(rc = {'figure.figsize':(11,4)})

#Initializing SQL engine
engine = create_engine('sqlite:///./nh19_sentiment.db')

#Reading dataframes
sentiment_df = pd.read_sql("""select * from comments""",engine)
embassy_df = sentiment_df[sentiment_df['page']=='USEmbassyGeorgetown']
southcom_df=sentiment_df[sentiment_df['page']=='southcom']
nh_df=sentiment_df[sentiment_df['page']=='AFSOUTHNewHorizons']

#Set working dataframe
working_df = embassy_df
working_df = working_df[working_df['created_time']>'2019-01-01']

#Converting and batching in weeks
working_df.created_time = pd.to_datetime(working_df.created_time)
#working_df['time_series'] = working_df['created_time'].dt.to_period('W').apply(lambda r: r.start_time)
#working_df = working_df.set_index('created_time')
#working_df = working_df.groupby(['time_series'], as_index = False).mean()
working_df = working_df.groupby(pd.Grouper(key='created_time', freq='W-MON')).mean().reset_index()
#print(working_df)
#
##Create data frame from non-null rows
plot_df = working_df.fillna(0)
##plot_df = working_df[pd.notnull(working_df['compound'])]
#print(plot_df['time_series'])
#plot_df['time_series'] = plot_df['time_series'].dt.strftime("%Y %m %d")
plot_df['created_time'] = plot_df['created_time'].dt.strftime("%Y %m %d")
#print(plot_df)
#plot_df.plot.scatter('created_time','compound')
#sns.catplot(x='time_series',y='compound',data=plot_df)
ax = sns.pointplot(x='created_time',y='compound',data=plot_df, s=10, color='blue')
ax.grid(True)
plt.title('Sentiment Over Time', fontsize = 28)
plt.xticks(rotation=45,ha = 'right', rotation_mode = 'anchor')
plt.show()
