import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sqlalchemy import create_engine
from collections import Counter
from tokenizer import *

import matplotlib.pyplot as plt

#allow use of arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Specify whether to use all facebook pages, only US facebook pages, or only Guyana faebook pages. Default is us", choices=['all','us','guy'], default='us')
arg_parser.add_argument("--start_date", help="Earliest date for posts/comments in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--end_date", help="Latest date for posts/comments in format YYYY-MM-DD. Default is 2019-06-22.", default="2019-06-22")
args = arg_parser.parse_args()

#stop words for both types of pages
us_stop_words = ["Guyana","Guyanese","USA","United","States","US","U.S.","America","Military","Ambassador","Air","Force"]
guy_stop_words = ["Guyana","Guyanese"]

#parts of speech
allowed_pos = ['NOUN', 'VERB', 'PROPN']

#stop lemmas for guyana pages
us_stop_lemmas = ["de","lo","el","por","la","que","los","en","es","con"]
guy_stop_lemmas = ["say", "man", "people","know","time","need","want","go","get","year","word","guyana","like","good","thing","come","let","think","look","right","day","lol"]

#define and instantiate tokenizers
us_tokenizer = Tokenizer(stop_words=us_stop_words, case_sensitive=False, stop_lemmas=us_stop_lemmas, allowed_pos=allowed_pos, lemma_token=False)
guy_tokenizer = Tokenizer(stop_words=guy_stop_words, case_sensitive=False, stop_lemmas=guy_stop_lemmas, allowed_pos=allowed_pos, lemma_token=False)

#pages used for facebook pull
if args.pages == 'all':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','dpiguyana','AFSouthern','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']
elif args.pages == 'us':
    page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','AFSouthern']
elif args.pages == 'guy':
    page_ids = ['dpiguyana','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle','gytimes','newsroomgy']

#lookup for detemining correct tokenizer 
page_to_tokenizer = {
    'AFSOUTHNewHorizons': us_tokenizer,
    'USEmbassyGeorgetown': us_tokenizer,
    'southcom': us_tokenizer,
    'dpiguyana': guy_tokenizer,
    'AFSouthern': us_tokenizer,
    'NewsSourceGuyana': guy_tokenizer,
    '655452691211411': guy_tokenizer,
    'kaieteurnewsonline': guy_tokenizer,
    'demwaves': guy_tokenizer,
    'CapitolNewsGY': guy_tokenizer,
    'PrimeNewsGuyana': guy_tokenizer,
    'INews.Guyana': guy_tokenizer,
    'stabroeknews': guy_tokenizer,
    'NCNGuyanaNews': guy_tokenizer,
    'dailynewsguyana': guy_tokenizer,
    'actionnewsguyana': guy_tokenizer,
    'gychronicle': guy_tokenizer,
    'gytimes': guy_tokenizer,
    'newsroomgy': guy_tokenizer
}


engine = create_engine('sqlite:///./nh19_fb.db')

#get all comments or posts for a date up to most recent post/comment 
raw_df = pd.read_sql("""select * from {0} where created_time >= '{1}' and created_time <= '{2}'""".format(args.type, args.start_date, args.end_date),engine)

#find date of most recent_comment
relevant_pages = raw_df[raw_df['page'].isin(page_ids)]
most_recent_date = relevant_pages.created_time.max()[:10]

#filter for page and store list of comments in 
page_comments = {} 
for pageid in page_ids:
    tmp_df = raw_df[raw_df['page'] == pageid]
    page_comments[pageid] = tmp_df[tmp_df['message'].notnull()].message.to_list()

#get list of tokens
page_tokens = {}
for page in page_comments: 
    print('Tokenizing {0} {1} from {2}'.format(len(page_comments[page]), args.type, page))
    page_tokens[page] = page_to_tokenizer[page].tokenize(page_comments[page], return_docs=False)

#combine all tokens for overall chart
all_tokens = [token for page in page_tokens for token in page_tokens[page]]

#convert tokens to string for wordcloud
page_string = {}
for page in page_tokens:
    page_string[page] = " ".join(page_tokens[page])

#combine all tokens for overall chart
all_string = " ".join(all_tokens) 

#keep collocations to show bigrams (increases understanding) and remove plurals (bin a little better)
page_wordclouds = {}
for page in page_string:
    page_wordclouds[page] = WordCloud(width=1000, height=600, background_color="white", collocations=False, normalize_plurals=True).generate(page_string[page])

#create wordcloud for overall chart
overall_wordcloud = WordCloud(width=1000, height=600, background_color="white", collocations=False, normalize_plurals=True).generate(all_string)

#generate all wordclouds
for i,page in enumerate(page_wordclouds):
    #wordcloud
    plt.figure(i+1)
    plt.imshow(page_wordclouds[page], interpolation='bilinear')
    plt.axis("off")
    plt.title('{0} Word Cloud'.format(page), fontsize = 20)
    plt.savefig('./wordclouds/{0}_{1}_wordcloud_{2}_to_{3}.png'.format(page,args.type,args.start_date,args.end_date))


plt.figure(len(page_ids)+1)
plt.imshow(overall_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Overall {0} Word Cloud, {1} - {2}".format(args.pages.upper(), args.start_date, args.end_date), fontsize = 20)
plt.savefig('./wordclouds/overall_{0}_{1}_freq_{2}_to_{3}.png'.format(args.pages,args.type,args.start_date,args.end_date))

plt.show()

