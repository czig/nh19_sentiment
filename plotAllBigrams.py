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
import math
from sqlalchemy import create_engine
from tokenizer import *
import json
from matplotlib.colors import rgb2hex

#seaborn settings
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#allow use of arguments
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", help="Specify whether to use posts or comments. Default is comments.", choices=['posts','comments'], default='comments')
arg_parser.add_argument("--pages", help="Specify whether to use all facebook pages, only US facebook pages, or only Guyana faebook pages. Default is us", choices=['all','us','guy'], default='us')
arg_parser.add_argument("--date", help="Earliest date for posts/comments in format YYYY-MM-DD. Default is 2019-04-01.", default="2019-04-01")
arg_parser.add_argument("--num_bigrams", help="Number of bigrams to show in barchart and graph. Default is 40.", type=int, default=40)
arg_parser.add_argument("--ignore", help="Ignore warnings", action="store_true")
args = arg_parser.parse_args()

#read off input values
if args.ignore:
    print('Ignoring all warnings...')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Using %s for analysis" % args.type)
print("Using start_date of: ",args.date)

#stop words for both types of pages
us_stop_words = ["Guyana","Guyanese","USA","United","States","US","U.S.","America","Military","Ambassador","Air","Force"]
guy_stop_words = ["Guyana","Guyanese"]

#stop lemmas
us_stop_lemmas = ["en","la","de","desde","los","eso","es"]

#define and instantiate tokenizers
us_tokenizer = Tokenizer(stop_words=us_stop_words, case_sensitive=False, stop_lemmas=us_stop_lemmas, lemma_token=False, lower_token=True)
guy_tokenizer = Tokenizer(stop_words=guy_stop_words, case_sensitive=False, lemma_token=False, lower_token=True)

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

#tokenize, create bigram counts, only keep most common bigrams and bigrams that show up at least a certain amount of times
page_bigram_dict = {}
page_tokens = {}
for page in page_comments: 
    print('Tokenizing {0} {1} from {2}'.format(len(page_comments[page]), args.type, page))
    tokens = page_to_tokenizer[page].tokenize(page_comments[page], return_docs=False)
    page_tokens[page] = tokens
    bigrams = list(nltk.bigrams(tokens))
    #keep top n bigrams
    bigram_count = collections.Counter(bigrams).most_common(args.num_bigrams)
    #make dictionary with bigram as key and count as value; only keep bigrams that appear more than once
    bigram_dict = {item[0]: item[1] for item in bigram_count if item[1] > 1}
    page_bigram_dict[page] = bigram_dict

#get bigram counts for overall chart
all_tokens = [token for page in page_tokens for token in page_tokens[page]]
all_bigrams = list(nltk.bigrams(all_tokens))
all_bigram_counts = collections.Counter(all_bigrams).most_common(args.num_bigrams)
all_bigram_dict = {item[0]: item[1] for item in all_bigram_counts if item[1] > 1}

#barchart for most common bigrams
for i,page in enumerate(page_bigram_dict):
    plt.figure(i)
    keys = [bigram[0] + '_' + bigram[1] for bigram in page_bigram_dict[page].keys()]
    plt.bar(keys,list(page_bigram_dict[page].values()))
    plt.xticks(rotation=40, ha='right')
    plt.subplots_adjust(bottom=0.4)
    plt.xlabel("Bigram")
    plt.ylabel("Frequency")
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.title('{0} Bigrams, {1} - {2}'.format(page, args.date, most_recent_date), fontsize = 20)
    plt.savefig('./bigramCharts/{0}_{1}_bigramBar_{2}_to_{3}.png'.format(page,args.type,args.date,most_recent_date))

#barchart for most common bigrams overall
plt.figure(len(page_bigram_dict))
all_keys = [bigram[0] + '_' + bigram[1] for bigram in all_bigram_dict.keys()]
plt.bar(all_keys,list(all_bigram_dict.values()))
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.4)
plt.xlabel("Bigram")
plt.ylabel("Frequency")
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.title('Overall {0} Bigrams, {1} - {2}'.format(args.pages.upper(), args.date, most_recent_date), fontsize = 20)
plt.savefig('./bigramCharts/overall_{0}_{1}_bigramBar_{2}_to_{3}.png'.format(args.pages,args.type,args.date,most_recent_date))

#set scaling factor for how spread out bigrams are (smaller makes more spread)
scale_factor = 5
weight_mult = args.num_bigrams/scale_factor

#Graphing network digram
for page in page_bigram_dict:
    G = nx.Graph()
    node_dict = {} 

    #add all edges (bigrams) to graph
    for k,v in page_bigram_dict[page].items():
        G.add_edge(k[0],k[1], weight=(v))
        #populate word frequency from bigrams alone
        if k[0] in node_dict.keys():
            node_dict[k[0]] += v
        else:
            node_dict[k[0]] = v
        if k[1] in node_dict.keys():
            node_dict[k[1]] += v
        else:
            node_dict[k[1]] = v

    max_weight = max([v for v in page_bigram_dict[page].values()])
    fig,ax = plt.subplots(figsize=(18,14))
    pos = nx.spring_layout(G,k=max(math.log(max_weight+1,10),1))

    #draw graph 
    nx.draw_networkx(G,pos,font_size=16,width=3,edge_color='grey',node_size=200,node_color=list(node_dict.values()),cmap=plt.cm.plasma,with_labels=False,ax=ax)
    plt.title('{0} Bigram Diagram, {1} - {2}'.format(page, args.date, most_recent_date))

    #assign color to each node based on node weight (number of bigram connections to each node)
    color_scale = plt.cm.plasma.__copy__()
    max_val = max(node_dict.values())
    node_colors = {node: rgb2hex(color_scale(node_dict[node]/max_val)) for node in node_dict}

    #get nodes and edges
    cyto_dict = nx.readwrite.json_graph.cytoscape_data(G)['elements']
    #add weight and color to each node
    for node in cyto_dict['nodes']:
        node['data']['weight'] = node_dict[node['data']['id']]
        node['data']['node_color'] = node_colors[node['data']['id']]
    #write json file to disk
    cyto_json_path = "./bigramCharts/cyto_json_{0}_{1}_{2}_to_{3}.json".format(page, args.type, args.date, most_recent_date)
    with open(cyto_json_path,"w") as cyto_file:
        cyto_file.write(json.dumps(cyto_dict))

    #turn off grid lines and both axes
    ax.axis('off')

    #labelling diagram
    for key, value in pos.items():
        x = value[0]
        y = value[1]+0.05
        ax.text(x,y,s=key,bbox=dict(facecolor='grey',edgecolor='black',alpha=0.1),horizontalalignment='center',fontsize=13)

    #create colorbar as legend for colormap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm._A = list(node_dict.values()) 
    cbar = plt.colorbar(sm)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Word Frequency', rotation=270)
    #save figure 
    plt.savefig('./bigramCharts/{0}_{1}_bigramGraph_{2}_to_{3}.png'.format(page,args.type,args.date,most_recent_date))


#Graphing for overall plot
G = nx.Graph()
node_dict = {} 

for k,v in all_bigram_dict.items():
    G.add_edge(k[0],k[1], weight=(v))
    if k[0] in node_dict.keys():
        node_dict[k[0]] += v
    else:
        node_dict[k[0]] = v
    if k[1] in node_dict.keys():
        node_dict[k[1]] += v
    else:
        node_dict[k[1]] = v

max_weight = max([v for v in all_bigram_dict.values()])
fig,ax = plt.subplots(figsize=(18,14))
pos = nx.spring_layout(G,scale=100,center=[50,50],k=1)

#draw graph
nx.draw_networkx(G,pos=pos,font_size=16,width=3,node_size=200,node_color=list(node_dict.values()),edge_color='grey',cmap=plt.cm.plasma, with_labels=False,ax=ax)
plt.title('Overall {0} Bigram Diagram, {1} - {2}'.format(args.pages.upper(), args.date, most_recent_date))

color_scale = plt.cm.plasma.__copy__()
max_val = max(node_dict.values())
node_colors = {node: rgb2hex(color_scale(node_dict[node]/max_val)) for node in node_dict}

cyto_dict = nx.readwrite.json_graph.cytoscape_data(G)['elements']
for node in cyto_dict['nodes']:
    node['data']['weight'] = node_dict[node['data']['id']]
    node['data']['node_color'] = node_colors[node['data']['id']]

cyto_json_path = "./bigramCharts/cyto_json_{0}_{1}_{2}_to_{3}.json".format(args.pages, args.type, args.date, most_recent_date)
with open(cyto_json_path,"w") as cyto_file:
    cyto_file.write(json.dumps(cyto_dict))
#turn off grid lines and both axes
ax.axis('off')

#labelling diagram
for key, value in pos.items():
    x = value[0]
    y = value[1] + 0.05
    ax.text(x,y,s=key,horizontalalignment='center',fontsize=13,bbox=dict(facecolor='grey',edgecolor='black',alpha=0.1))

#create colorbar as legend for colormap
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
sm._A = list(node_dict.values()) 
cbar = plt.colorbar(sm)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Word Frequency', rotation=270)
#save figure
plt.savefig('./bigramCharts/overall_{0}_{1}_bigramGraph_{2}_to_{3}.png'.format(args.pages,args.type,args.date,most_recent_date))

plt.show()

