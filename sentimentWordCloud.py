import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

engine = create_engine('sqlite:///./nh19_sentiment.db')

sentiment_df = pd.read_sql("""select * from CommentSentiment""",engine)

embassy_df = sentiment_df[sentiment_df['page'] == "USEmbassyGeorgetown"]
southcom_df = sentiment_df[sentiment_df['page'] == "southcom"]
nh_df = sentiment_df[sentiment_df['page'] == "AFSOUTHNewHorizons"]

text = " ".join(comment for comment in sentiment_df.message)
text2 = " ".join(comment for comment in embassy_df.message)
text3 = " ".join(comment for comment in southcom_df.message)
text4 = " ".join(comment for comment in nh_df.message)

stopwords = set(STOPWORDS)
stopwords.update(["Guyana","Guyanese","USA","United","States","US","America","Military","Ambassador"])
wordcloud = WordCloud(width=1000, height=600, stopwords=stopwords, background_color="white").generate(text)
wordcloud2 = WordCloud(width = 1000, height=600, stopwords = stopwords, background_color="white").generate(text2)
wordcloud3 = WordCloud(width = 1000, height=600, stopwords = stopwords, background_color="white").generate(text3)
wordcloud4 = WordCloud(width = 1000, height=600, stopwords = stopwords, background_color="white").generate(text4)

#plt.figure(0)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")

#plt.figure(1)
#plt.imshow(wordcloud1, interpolation='bilinear')
#plt.axis("off")

#plt.figure(2)
#plt.imshow(wordcloud2, interpolation='bilinear')
#plt.axis("off")

#plt.figure(3)
#plt.imshow(wordcloud3, interpolation='bilinear')
#plt.axis("off")
#plt.show()

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

plt.show()

