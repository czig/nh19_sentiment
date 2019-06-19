import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

engine = create_engine('sqlite:///./nh19_sentiment.db')

sentiment_df = pd.read_sql("""select * from CommentSentiment""",engine)

page_stats = sentiment_df.groupby("page")
print(page_stats.max())
page_stats.size().sort_values(ascending=False).plot.bar()
#page_stats.max().sort_values(by="compound",ascending=False)["compound"].plot.bar()
plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.xlabel("Facebook Page")
plt.ylabel("Number of Comments")
plt.title("Number of Comments Per Facebook Page")
plt.show()
