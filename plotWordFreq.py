import pandas as pd
import numpy as np
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

#connect to database
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
            elif token.pos_ == "PRON":
                continue
            else:
                #TODO: use lemma or not??
                all_tokens.append(token.lemma_)

    return all_tokens

#get list of tokens
all_tokens = tokenize(all_comments, parser)
embassy_tokens = tokenize(embassy_comments, parser)
southcom_tokens = tokenize(southcom_comments, parser)
nh_tokens = tokenize(nh_comments, parser)

#number of bars in each chart
num_words = 10

#find counts for top n most common words
all_freq = dict(Counter(all_tokens).most_common(num_words))
embassy_freq = dict(Counter(embassy_tokens).most_common(num_words))
southcom_freq = dict(Counter(southcom_tokens).most_common(num_words))
nh_freq = dict(Counter(nh_tokens).most_common(num_words))

#Word frequency plots
plt.figure(1)
plt.bar(all_freq.keys(), all_freq.values())
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Frequency")
plt.title('Overall Word Frequency', fontsize = 36)

plt.figure(2)
plt.bar(embassy_freq.keys(), embassy_freq.values())
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Frequency")
plt.title('Embassy Word Frequency', fontsize = 36)

plt.figure(3)
plt.bar(southcom_freq.keys(), southcom_freq.values())
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Frequency")
plt.title('SOUTHCOM Word Frequency', fontsize = 36)

plt.figure(4)
plt.bar(nh_freq.keys(), nh_freq.values())
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Frequency")
plt.title('NH Word Frequency', fontsize = 36)

plt.show()
