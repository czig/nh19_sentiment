import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import Counter
import matplotlib.pyplot as plt
import time

#allow use of arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--date", help="Earliest date for posts/comments in format YYYY-MM-DD", default="2019-04-01")
args = arg_parser.parse_args()

#import spacy
import spacy
parser = spacy.load('en_core_web_sm')
stop_words = ["Guyana","Guyanese"]
for word in stop_words:
    parser.vocab[word].is_stop = True
    parser.vocab[word.lower()].is_stop = True

#guyana pages from facebook pull
guyana_pages = ['dpiguyana','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']

#connect to database
engine = create_engine('sqlite:///./nh19_fb.db')

#get all comments or posts for a time period 
df = pd.read_sql("""select * from {0} where created_time > '{1}'""".format(args.type, args.date),engine)
guy_df = df[df['page'].isin(guyana_pages)]

#find date of most recent_comment
connection = engine.connect()
result = connection.execute("""select substr(max(created_time),1,10) from {0}""".format(args.type)) 
for row in result:
    most_recent_date = row[0]
connection.close()

#convert to list
all_comments = guy_df[guy_df['message'].notnull()].message.to_list()

#tokenize and clean posts
def tokenize(messages_list, parser):
    all_tokens = []
    for message in messages_list:
        #convert unicode punctuation to regular ascii punctuation 
        message = message.replace(chr(8216),"'")
        message = message.replace(chr(8217),"'")
        message = message.replace(chr(8218),",")
        message = message.replace(chr(8220),'"')
        message = message.replace(chr(8221),'"')
        message = message.replace(chr(8242),'`')
        message = message.replace(chr(8245),'`')
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
                #use lemma here to make charts more understandable/relevant
                all_tokens.append(token.lemma_)

    return all_tokens

#get list of tokens
print('tokenizing {0} comments'.format(len(all_comments)))
start_tokenize = time.time()
all_tokens = tokenize(all_comments, parser)
end_tokenize = time.time()
print('done tokenizing')
print('tokenizing time: ',end_tokenize - start_tokenize)

#number of bars in each chart
num_words = 20

#find counts for top n most common words
print('getting word frequencies')
start_freq = time.time()
all_freq = dict(Counter(all_tokens).most_common(num_words))
end_freq = time.time()
print('done with word frequencies')
print('freq time: ',end_freq - start_freq)

#Word frequency plots
plt.figure(1)
plt.bar(all_freq.keys(), all_freq.values())
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Frequency")
plt.title('Overall Guyana Word Frequency {0} - {1}'.format(args.date, most_recent_date), fontsize = 20)
plt.savefig('./wordFreqPlots/overallGuy_{0}_freq_{1}_to_{2}.png'.format(args.type,args.date,most_recent_date))

plt.show()
