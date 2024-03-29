import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

engine = create_engine('sqlite:///./nh19_fb.db')

embassy_df = pd.read_sql("""select * from comments
                              where page="USEmbassyGeorgetown" and created_time > "2019-04-01"
                         """,engine)

text = " ".join(comment for comment in embassy_df.message)

stopwords = set(STOPWORDS)
stopwords.update(["Guyana","Guyanese","USA","United","States","US","America"])
wordcloud = WordCloud(width=1000, height=600, stopwords=stopwords, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
