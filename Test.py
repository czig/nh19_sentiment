import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Initialize sentiment analysis
sid = SentimentIntensityAnalyzer()

#Open text file to write
file1 = open("Test1.txt","w")
Comments = [] #Initialize list of comments
sum_pos = 0 #sum of positive sentiment scores
sum_neg = 0 #sum of negative sentiment scores
sum_neu = 0 #sum of neutral sentiment scores
sum_compound = 0 #sum of compound sentiment scores
Comment_Count = 0 #counting total number of non-blank comments

#Open csv of comments
with open('Test.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        #Loop through rows in csv
        line_count = 0
        for row in csv_reader:

                #Printing column names
                if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1

                else:
                        alpha = sid.polarity_scores(row[8]) #Perform sentiment analysis
                       
                        #create dictionary containing comment content and sentiment scores. 
                        comment_dict = {'comment': row[8], 'sentiment': alpha}
                        #Can create additional entires for more sorting tags

                        file1.write(comment_dict['comment']+"\n")#Write comment content
                        file1.write(str(comment_dict['sentiment'])+"\n\n")#Write sentiment scores
                        Comments.append(comment_dict)#Add to list of comments
                       
                       line_count += 1#Next line

                        #Check if comment is not blank to add to count
                        if alpha['compound'] != 0:
                            Comment_Count += 1
                        elif alpha['pos'] != 0:
                            Comment_Count += 1
                        elif alpha['neg'] != 0:
                            Comment_Count += 1
                        elif alpha['neu'] != 0:
                            Comment_Count += 1

        #Summing sentiment scores
        for comment in Comments:
            sum_pos = sum_pos + comment['sentiment']['pos']
            sum_neg = sum_neg + comment['sentiment']['neg']
            sum_neu = sum_neu + comment['sentiment']['neu']
            sum_compound = sum_compound + comment['sentiment']['compound']

        #How many lines did we process
        print(f'Processed {line_count} lines.')

        #Average of sentiment scores
        file1.write('Overall positive sentiment: ' + str(sum_pos/Comment_Count)+"\n")
        file1.write('Overall negative sentiment: ' + str(sum_neg/Comment_Count)+"\n")
        file1.write('Overall neutral sentiment: ' + str(sum_neu/Comment_Count)+"\n")
        file1.write('Overall sentiment: ' + str(sum_compound/Comment_Count)+"\n")
file1.close()
