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

stopwords = set(STOPWORDS)
stopwords.update(["Guyana","Guyanese","USA","United","States","US","America","Military"])
wordcloud = WordCloud(width=1000, height=600, stopwords=stopwords, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
