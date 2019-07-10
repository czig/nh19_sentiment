import sqlalchemy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
from sqlalchemy import create_engine

#Reading database
engine = create_engine('sqlite:///./topic_logging.db')
coherence = pd.read_sql("""select pass, num_topics, iterations, total_passes, update_every, chunksize, metric_type, metric_val, test_name from topic_logs where (test_name='doe1' or test_name = 'doe2') and metric_type = 'Coherence'""", engine)
perplexity = pd.read_sql("""select pass, num_topics, iterations, total_passes, update_every, chunksize, metric_type, metric_val, test_name from topic_logs where (test_name = 'doe1' or test_name = 'doe2') and metric_type = 'Perplexity'""", engine)
convergence = pd.read_sql("""select pass, num_topics, iterations, total_passes, update_every, chunksize, metric_type, metric_val, test_name from topic_logs where (test_name = 'doe1' or test_name = 'doe2') and metric_type = 'Convergence'""", engine)
print(coherence)
def Regression(df):
	filtered = df[((df['pass'] == df['total_passes']-1)|(df['pass']==97))&(df['test_name']=='doe1')].reset_index()
	print('filtered = ')
	print(filtered)
	filtered2 = df[(df['pass'] == df['total_passes']-1)&(df['test_name'] == 'doe2')].reset_index()
	print('filtered 2 = ')
	print(filtered2)
	poop = pd.concat([filtered, filtered2])
	print(poop)
	poop = poop.drop(['index','pass','metric_type','test_name'],axis = 1)
	poop = poop.astype(float)
	#print(poop)
	#model = "(num_topics+total_passes+update_every+chunksize)**2+np.power(num_topics,2)+np.power(total_passes,2)+np.power(update_every,2)+np.power(chunksize,2)"
	model = "(num_topics+update_every+chunksize)**2+I(num_topics**2)+I(update_every**2)+I(chunksize**2)"
	lm = ols('metric_val~'+model, data=poop).fit()
	print(lm.summary())

Regression(convergence)






