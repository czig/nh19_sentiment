import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

engine = create_engine('sqlite:///./nh19_sentiment.db')

sentiment_df = pd.read_sql("""select * from comments where created_time > '2019-01-01'""",engine)

page_stats = sentiment_df.groupby("page")
print(page_stats.max())
plt.figure(1)
page_stats.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Facebook Page", fontsize = 20)
plt.ylabel("Number of Comments", fontsize = 20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Number of Comments Per Facebook Page", fontsize = 24)

plt.figure(2)
page_stats.mean().sort_values(by="compound",ascending=False)["compound"].plot.bar()
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Facebook Page", fontsize = 20)
plt.ylabel("Compound Sentiment Score", fontsize = 20)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.title("Average Compound Sentiment Score Per Facebook Page", fontsize = 24)

plt.show()
