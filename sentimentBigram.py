#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
import re
import networkx as nx
import warnings
import sqlalchemy
import string
from sqlalchemy import create_engine
import re

warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#set stop word conditions
stop_words = set(stopwords.words('english'))

#reading SQL databases
engine = create_engine('sqlite:///./nh19_sentiment.db')
sentiment_df = pd.read_sql("""select * from Commentsentiment""", engine)
embassy_df = sentiment_df[sentiment_df['page'] == "USEmbassyGeorgetown"]
southcom_df = sentiment_df[sentiment_df['page']=="southcom"]
nh_df = sentiment_df[sentiment_df['page']=='AFSOUTHNewHorizons']

#generating/cleaning text
text_list = sentiment_df.message.tolist()
text_np = [re.sub(r'[^\w\s]','',comment) for comment in text_list] #deleting punctuation
text = [comment.lower().split() for comment in text_np] #lower case all words, split into list of lists
text_nsw = [[word for word in comment_words if not word in stop_words] for comment_words in text] #removing stop words

#Bigrams
terms_bigram = [list(bigrams(comments)) for comments in text_nsw] #creating bigrams

bigrams = list(itertools.chain(*terms_bigram))
bigram_counts = collections.Counter(bigrams)#counting how many of each bigram shows up
bigram_df = pd.DataFrame(bigram_counts.most_common(20), columns = ['bigram', 'count']) #creating dataframe of x most bigrams
print(bigram_df) #show the dataframe

#Graphing network digram
d = bigram_df.set_index('bigram').T.to_dict('records')
G = nx.Graph()

for k,v in d[0].items():
	G.add_edge(k[0],k[1], weight=(v*10))

fig,ax = plt.subplots(figsize=(10,8))
pos = nx.spring_layout(G,k=1)

nx.draw_networkx(G,pos,font_size=16,width=3,edge_color='grey',node_color='purple',with_labels=False,ax=ax)

#labelling diagram
for key, value in pos.items():
	x,y = value[0]+.05, value[1]+0.01
	ax.text(x,y,s=key,bbox=dict(facecolor='red',alpha=0.25),horizontalalignment='center',fontsize=13)
plt.show()
