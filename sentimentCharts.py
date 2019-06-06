import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

engine = create_engine('sqlite:///./nh19_sentiment.db')

sentiment_df = pd.read_sql("""select * from CommentSentiment""",engine)

page_stats = sentiment_df.groupby("page")
print(page_stats.max())
plt.figure(figsize=(15,10))
page_stats.size().sort_values(ascending=False).plot.bar()
#page_stats.max().sort_values(by="compound",ascending=False)["compound"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Page")
plt.ylabel("Number of Comments")
plt.show()
