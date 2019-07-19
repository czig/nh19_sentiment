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
stop_words.update(['ai','ca',"n't",'wo','..','...','that',"'s"])
#reading SQL databases
engine = create_engine('sqlite:///./nh19_sentiment.db')
sentiment_df = pd.read_sql("""select * from Commentsentiment""", engine)
embassy_df = sentiment_df[sentiment_df['page'] == "USEmbassyGeorgetown"]
southcom_df = sentiment_df[sentiment_df['page']=="southcom"]
nh_dr = sentiment_df[sentiment_df['page']=='AFSOUTHNewHorizons']

df = sentiment_df

#generating/cleaning text
#text_list = sentiment_df.message.tolist()
#text_np = [re.sub(r'[^\w\s]','',comment) for comment in text_list] #deleting punctuation
#text = [comment.lower().split() for comment in text_np] #lower case all words, split into list of lists
#text_nsw = [[word for word in comment_words if not word in stop_words] for comment_words in text] #removing stop words

text = " ".join(comment for comment in df.message)
words = nltk.word_tokenize(text)
words = [word for word in words if len(word)>1]
words = [word for word in words if word not in stop_words]
words = [word.lower() for word in words]

#Bigrams
terms_bigram = list(nltk.bigrams(words)) #creating bigrams
#print(terms_bigram)
#bigrams = list(itertools.chain(*terms_bigram))
bigram_counts = collections.Counter(terms_bigram)#counting how many of each bigram shows up
print('bigram df')
print(bigram_counts)
bigram_df = pd.DataFrame(bigram_counts.most_common(1000), columns = ['bigram', 'count']) #creating dataframe of x most bigrams
print('first bigram df')
print(bigram_df)
bigram_df = bigram_df[bigram_df['count'] > 750]
print('count filtered bigram df')
print(bigram_df) #show the dataframe

#Graphing network digram
d = bigram_df.set_index('bigram').T.to_dict('records')
print('bigram df to dict')
print(d)
G = nx.Graph()

for k,v in d[0].items():
	G.add_edge(k[0],k[1], weight=(v*10))

fig,ax = plt.subplots(figsize=(10,8))
pos = nx.spring_layout(G,k=1)

nx.draw_networkx(G,pos,font_size=16,width=3,edge_color='grey',node_color='purple',with_labels=False,ax=ax)
plt.title('Overall Facebook Bigram Diagram')

#labelling diagram
# for key, value in pos.items():
	# x,y = value[0]+.005, value[1]+0.001
	# ax.text(x,y,s=key,bbox=dict(facecolor='red',alpha=0.25),horizontalalignment='center',fontsize=13)
plt.show()
