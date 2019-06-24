import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sqlalchemy import create_engine
from collections import Counter

import matplotlib.pyplot as plt

#import spacy
import spacy
parser = spacy.load('en_core_web_sm')
stop_words = ["Guyana","Guyanese","USA","United","States","US","America","Military","Ambassador"]
for word in stop_words:
    parser.vocab[word].is_stop = True
    parser.vocab[word.lower()].is_stop = True

engine = create_engine('sqlite:///./nh19_sentiment.db')

#get all comments for this year
sentiment_df = pd.read_sql("""select * from CommentSentiment where created_time > '2019-01-01'""",engine)

#filter for page
embassy_df = sentiment_df[sentiment_df['page'] == "USEmbassyGeorgetown"]
southcom_df = sentiment_df[sentiment_df['page'] == "southcom"]
nh_df = sentiment_df[sentiment_df['page'] == "AFSOUTHNewHorizons"]

#convert to list
all_comments = sentiment_df[sentiment_df['message'].notnull()].message.to_list()
embassy_comments = embassy_df[embassy_df['message'].notnull()].message.to_list()
southcom_comments = southcom_df[southcom_df['message'].notnull()].message.to_list()
nh_comments = nh_df[nh_df['message'].notnull()].message.to_list()

#tokenize and clean posts
def tokenize(messages_list, parser):
    all_tokens = []
    for message in messages_list:
        #unicode for 's (right apostrophie followed by s)
        possessive_substr = chr(8217) + 's'
        message_tokens = parser(message)
        #iterate through all tokens in each post
        for token in message_tokens:
            #remove space
            if token.orth_.isspace():
                continue
            #remove punctuation
            elif token.is_punct:
                continue
            #remove urls
            elif token.like_url:
                continue
            #remove emails
            elif token.like_email:
                continue
            #remove stop words
            elif token.is_stop:
                continue
            #remove 's
            elif token.text.find(possessive_substr) > -1:
                continue
            #remove single letters
            elif len(token.text) < 2:
                continue
            #remove pronouns
            elif token.lemma_ == "-PRON-":
                continue
            else:
                #TODO: use lemma or not??
                all_tokens.append(token.lemma_)

    return all_tokens

all_tokens = tokenize(all_comments, parser)
embassy_tokens = tokenize(embassy_comments, parser)
southcom_tokens = tokenize(southcom_comments, parser)
nh_tokens = tokenize(nh_comments, parser)

all_text = " ".join(all_tokens)
embassy_text = " ".join(embassy_tokens)
southcom_text = " ".join(southcom_tokens)
nh_text = " ".join(nh_tokens)

num_words = 10
all_freq = dict(Counter(all_tokens).most_common(num_words))
embassy_freq = dict(Counter(embassy_tokens).most_common(num_words))
southcom_freq = dict(Counter(southcom_tokens).most_common(num_words))
nh_freq = dict(Counter(nh_tokens).most_common(num_words))

#TODO: keep collocations? normalize plurals or not??
wordcloud = WordCloud(width=1000, height=600, background_color="white", collocations=False, normalize_plurals=False).generate(all_text)
wordcloud2 = WordCloud(width = 1000, height=600, background_color="white", collocations=False, normalize_plurals=False).generate(embassy_text)
wordcloud3 = WordCloud(width = 1000, height=600, background_color="white", collocations=False, normalize_plurals=False).generate(southcom_text)
wordcloud4 = WordCloud(width = 1000, height=600, background_color="white", collocations=False, normalize_plurals=False).generate(nh_text)

#create subplot with all wordcloud
fig,axs = plt.subplots(2,2,constrained_layout=True)
axs[0,0].imshow(wordcloud, interpolation='bilinear')
axs[0,0].axis("off")
axs[0,0].set_title('Overall Word Cloud', fontsize = 28)

axs[0,1].imshow(wordcloud2, interpolation='bilinear')
axs[0,1].axis("off")
axs[0,1].set_title('Embassy Word Cloud', fontsize = 28)

axs[1,0].imshow(wordcloud3, interpolation='bilinear')
axs[1,0].axis("off")
axs[1,0].set_title('SOUTHCOM Word Cloud', fontsize = 28)

axs[1,1].imshow(wordcloud4, interpolation='bilinear')
axs[1,1].axis("off")
axs[1,1].set_title('NH Facebook Word Cloud', fontsize=28)

#wordcloud
plt.figure(2)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Overall Word Cloud', fontsize = 36)

plt.figure(3)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.title('Embassy Word Cloud', fontsize = 36)

plt.figure(4)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.title('SOUTHCOM Word Cloud', fontsize = 36)

plt.figure(5)
plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis("off")
plt.title('NH Facebook Word Cloud', fontsize = 36)

#Word frequency plots
plt.figure(6)
plt.bar(all_freq.keys(), all_freq.values())
plt.ylabel("Frequency")
plt.title('Overall Word Frequency', fontsize = 36)

plt.figure(7)
plt.bar(embassy_freq.keys(), embassy_freq.values())
plt.ylabel("Frequency")
plt.title('Embassy Word Frequency', fontsize = 36)

plt.figure(8)
plt.bar(southcom_freq.keys(), southcom_freq.values())
plt.ylabel("Frequency")
plt.title('SOUTHCOM Word Frequency', fontsize = 36)

plt.figure(9)
plt.bar(nh_freq.keys(), nh_freq.values())
plt.ylabel("Frequency")
plt.title('NH Word Frequency', fontsize = 36)

plt.show()

