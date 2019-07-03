import logging
import traceback
from sqlalchemy import create_engine
import re
#set callbacks
class dbLogger(logging.Handler):
    def set_values(self, num_topics, iterations, total_passes, update_every, chunksize, alpha, beta, 
                 message_type, start_date, doc_size):
        self.num_topics = num_topics
        self.iterations = iterations
        self.total_passes = total_passes
        self.update_every = update_every
        self.chunksize = chunksize
        self.alpha = alpha
        self.beta = beta
        self.message_type = message_type
        self.start_date = start_date
        self.doc_size = doc_size
        #create logging database
        self.log_conn = create_engine('sqlite:///./topic_logging.db')
        #TODO: change insert and table definition
        self.insert_string = """insert into topic_logs
                            (pass, num_topics, iterations, total_passes, update_every, chunksize, alpha, beta, 
                             message_type, start_date, doc_size, metric_type, metric_val)
                           values 
                            ({0}, {1}, {2}, {3}, {4}, {5}, '{6}', '{7}', '{8}', '{9}', {10}, '{11}', '{12}')"""

    def emit(self, record):
        trace = None 
        exc = record.__dict__['exc_info']
        if exc:
            trace = traceback.format_exc()
        record_dict = record.__dict__
        message = record_dict['message']
        if record_dict['module'] == 'callbacks':
            epoch = int(re.search(r'(?<=Epoch\s)\d+',message).group(0))
            metric_type = re.search(r'(?<=:\s)\w+(?=\sestimate:)',message).group(0)
            if metric_type == 'Diff':
                message = re.sub("\s\s+"," ", message.replace('\n',""))
                metric_val = re.search(r'(?<=:\s)\[.*\]',message).group(0)
            else:
                metric_val = re.search(r'(?<=:\s)[\d\.]+',message).group(0)
            #insert information into log TODO: change columns (need metric type and metric name) 
            self.log_conn.execute(self.insert_string.format(epoch, self.num_topics, self.iterations, self.total_passes, self.update_every, self.chunksize, self.alpha, self.beta, self.message_type, self.start_date, self.doc_size, metric_type, metric_val))

