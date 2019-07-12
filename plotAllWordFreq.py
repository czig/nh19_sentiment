import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import Counter
import matplotlib.pyplot as plt

#allow use of arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Specify whether to use all facebook pages, only US facebook pages, or only Guyana faebook pages", choices=['all','us','guy'], default='us')
arg_parser.add_argument("--date", help="Earliest date for posts/comments in format YYYY-MM-DD", default="2019-04-01")
args = arg_parser.parse_args()

#import spacy and make separate parsers for US pages and Guyana pages
import spacy
us_parser = spacy.load('en_core_web_sm')
guy_parser = spacy.load('en_core_web_sm')
us_stop_words = ["Guyana","Guyanese","USA","United","States","US","America","Military","Ambassador"]
guy_stop_words = ["Guyana","Guyanese"]
for word in us_stop_words:
    us_parser.vocab[word].is_stop = True
    us_parser.vocab[word.lower()].is_stop = True
for word in guy_stop_words:
    guy_parser.vocab[word].is_stop = True
    guy_parser.vocab[word.lower()].is_stop = True

#pages used for facebook pull
if args.pages == 'all':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','dpiguyana','AFSouthern','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']
elif args.pages == 'us':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','AFSouthern']
elif args.pages == 'guy':
    page_ids = ['dpiguyana','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']

#lookup for detemining correct parser
page_to_parser = {
    'AFSOUTHNewHorizons': us_parser,
    'USEmbassyGeorgetown': us_parser,
    'southcom': us_parser,
    'dpiguyana': guy_parser,
    'AFSouthern': us_parser,
    'NewsSourceGuyana': guy_parser,
    '655452691211411': guy_parser,
    'kaieteurnewsonline': guy_parser,
    'demwaves': guy_parser,
    'CapitolNewsGY': guy_parser,
    'PrimeNewsGuyana': guy_parser,
    'INews.Guyana': guy_parser,
    'stabroeknews': guy_parser,
    'NCNGuyanaNews': guy_parser,
    'dailynewsguyana': guy_parser,
    'actionnewsguyana': guy_parser,
    'gychronicle': guy_parser,
    'gytimes': guy_parser,
    'newsroomgy': guy_parser
}

#connect to database
engine = create_engine('sqlite:///./nh19_fb.db')

#get all comments for this year
raw_df = pd.read_sql("""select * from {0} where created_time > '{1}'""".format(args.type, args.date),engine)

#find date of most recent_comment
relevant_pages = raw_df[raw_df['page'].isin(page_ids)]
most_recent_date = relevant_pages.created_time.max()[:10]

#filter for page and store list of comments in 
page_comments = {} 
for pageid in page_ids:
    tmp_df = raw_df[raw_df['page'] == pageid]
    page_comments[pageid] = tmp_df[tmp_df['message'].notnull()].message.to_list()

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
page_tokens = {}
for page in page_comments: 
    print('Tokenizing {0} {1} from {2}'.format(len(page_comments[page]), args.type, page))
    page_tokens[page] = tokenize(page_comments[page], page_to_parser[page])
    
#number of bars in each chart
num_words = 20

#find counts for top n most common words
page_word_freq = {} 
for page in page_tokens:
    page_word_freq[page] = dict(Counter(page_tokens[page]).most_common(num_words))

#Word frequency plots
for i,page in enumerate(page_word_freq):
    plt.figure(i+1)
    plt.bar(page_word_freq[page].keys(), page_word_freq[page].values())
    plt.xticks(rotation=40, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("Frequency")
    plt.title('{0} Word Frequency, {1} - {2}'.format(page, args.date, most_recent_date), fontsize = 20)
    plt.savefig('./wordFreqPlots/{0}_{1}_freq_{2}_to_{3}.png'.format(page,args.type,args.date,most_recent_date))

plt.show()
