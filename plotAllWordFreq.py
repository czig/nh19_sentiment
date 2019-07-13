import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import Counter
import matplotlib.pyplot as plt
from tokenizer import *

#allow use of arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Specify whether to use all facebook pages, only US facebook pages, or only Guyana faebook pages. Default is us", choices=['all','us','guy'], default='us')
arg_parser.add_argument("--date", help="Earliest date for posts/comments in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
args = arg_parser.parse_args()

#stop words for both types of pages
us_stop_words = ["Guyana","Guyanese","USA","United","States","US","America","Military","Ambassador"]
guy_stop_words = ["Guyana","Guyanese"]

#define and instantiate tokenizers
us_tokenizer = Tokenizer(stop_words=us_stop_words, case_sensitive=False, remove_pos=["PRON"])
guy_tokenizer = Tokenizer(stop_words=guy_stop_words, case_sensitive=False, remove_pos=["PRON"])

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


#get list of tokens
page_tokens = {}
for page in page_comments: 
    print('Tokenizing {0} {1} from {2}'.format(len(page_comments[page]), args.type, page))
    page_tokens[page] = page_to_tokenizer[page].tokenize(page_comments[page], return_docs=False)

#combine all tokens for overall chart
all_tokens = [token for page in page_tokens for token in page_tokens[page]]
    
#number of bars in each chart
num_words = 20

#find counts for top n most common words
page_word_freq = {} 
for page in page_tokens:
    page_word_freq[page] = dict(Counter(page_tokens[page]).most_common(num_words))

#get overall word count for overall chart
all_word_freq = dict(Counter(all_tokens).most_common(num_words))

#Word frequency plots
for i,page in enumerate(page_word_freq):
    plt.figure(i+1)
    plt.bar(page_word_freq[page].keys(), page_word_freq[page].values())
    plt.xticks(rotation=40, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("Frequency")
    plt.title('{0} Word Frequency, {1} - {2}'.format(page, args.date, most_recent_date), fontsize = 20)
    plt.savefig('./wordFreqPlots/{0}_{1}_freq_{2}_to_{3}.png'.format(page,args.type,args.date,most_recent_date))

#overall word frequency plot
plt.figure(len(page_word_freq)+1)
plt.bar(all_word_freq.keys(), all_word_freq.values())
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.ylabel("Frequency")
plt.title('Overall {0} Word Frequency, {1} - {2}'.format(args.pages.upper(), args.date, most_recent_date), fontsize = 20)
plt.savefig('./wordFreqPlots/overall_{0}_{1}_freq_{2}_to_{3}.png'.format(args.pages,args.type,args.date,most_recent_date))

plt.show()
