import facebook, urllib3, requests
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///./nh19_fb.db')

page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom']
access_token = 'EAAI4BG12pyIBADCni60YQaBwZCsTP3lki7dY73ZCn8YZAUT3FrwNkC6iRpP7qu2palZBMGnFjrEG8RIyPhUYQ4gjZBnPNKYRFdskfny5dZAZCxzEZA7OxnmwyoV0NQ1bPO9RedcJYrPMyjbbh7FfAPDyIQiZB35th5uRpPH1s9s2PXZCaEJ2MChQYB'

graph = facebook.GraphAPI(access_token=access_token,version=3.1)

def get_all_reactions(object_id):
    likes = graph.request(object_id + '?fields=reactions.type(LIKE).limit(0).summary(total_count)')['reactions']['summary']['total_count']
    loves = graph.request(object_id + '?fields=reactions.type(LOVE).limit(0).summary(total_count)')['reactions']['summary']['total_count']
    wows = graph.request(object_id + '?fields=reactions.type(WOW).limit(0).summary(total_count)')['reactions']['summary']['total_count']
    sads = graph.request(object_id + '?fields=reactions.type(SAD).limit(0).summary(total_count)')['reactions']['summary']['total_count']
    angrys = graph.request(object_id + '?fields=reactions.type(ANGRY).limit(0).summary(total_count)')['reactions']['summary']['total_count']
    return likes, loves, wows, sads, angrys

def handle_message_tags(fb_object,parent_type,tag_list):
    if 'message_tags' in fb_object:
        for tag in fb_object['message_tags']:
            tag['parent_id'] = fb_object[u'id']
            tag['parent_type'] = parent_type 
            tag.pop('length',None)
            tag.pop('offset',None)
            tag_list.append(tag)
        #remove message tags because no longer needed on comment 
        return fb_object.pop('message_tags',None)


def pull_data(page_id):
    #initialize
    feed_gen = graph.get_all_connections(id=page_id, connection_name='feed',fields="id,created_time,message,message_tags,parent_id,shares")
    feed = []
    all_comments = []
    all_message_tags = []
    #pull post data
    for post in feed_gen:
        if post[u'created_time'] > '2018-08-01':
            #handle information relating to post specifically
            print(post[u'created_time'])
            likes,loves,wows,sads,angrys = get_all_reactions(post[u'id'])
            post['likes'] = likes
            post['loves'] = loves
            post['wows'] = wows
            post['sads'] = sads
            post['angrys'] = angrys
            post[u'shares'] = post['shares']['count'] if 'shares' in post else 0
            post['page'] = page_id

            #handle message tags
            handle_message_tags(post,'post',all_message_tags)

            feed.append(post)

            #handle comments
            comments = graph.get_connections(id=post[u'id'], connection_name='comments', fields="id,message,like_count,message_tags,parent,created_time,comment_count")[u'data']
            for comment in comments:
                #keep track of parent and master page
                comment[u'parent'] = comment[u'parent'][u'id'] if u'parent' in comment else ""
                comment['parent_id'] = post[u'id']
                comment['parent_type'] = 'post'
                comment['page'] = page_id
                #get comment reactions
                likes,loves,wows,sads,angrys = get_all_reactions(comment[u'id'])
                comment['likes'] = likes
                comment['loves'] = loves
                comment['wows'] = wows
                comment['sads'] = sads
                comment['angrys'] = angrys
                #handle nested comments (if any exist)
                if comment[u'comment_count'] > 0:
                    nested_comments = graph.get_connections(id=comment[u'id'], connection_name='comments', fields="id,message,like_count,message_tags,parent,created_time,comment_count")[u'data'] 
                    for nested_comment in nested_comments:
                        nested_comment[u'parent'] = nested_comment[u'parent'][u'id'] if u'parent' in nested_comment else ""
                        nested_comment['parent_id'] = comment[u'id']
                        nested_comment['parent_type'] = 'comment'
                        nested_comment['page'] = page_id
                        #get comment reactions
                        likes,loves,wows,sads,angrys = get_all_reactions(nested_comment[u'id'])
                        nested_comment['likes'] = likes
                        nested_comment['loves'] = loves
                        nested_comment['wows'] = wows
                        nested_comment['sads'] = sads
                        nested_comment['angrys'] = angrys
                        # handle message tags
                        handle_message_tags(nested_comment,'comment',all_message_tags)

                        all_comments.append(nested_comment)

                # handle message tags
                handle_message_tags(comment,'comment',all_message_tags)

                all_comments.append(comment)

        else:
            break

    #make dataframe
    feed_df = pd.DataFrame(feed)
    comments_df = pd.DataFrame(all_comments)
    tags_df = pd.DataFrame(all_message_tags)
    
    return feed_df,comments_df,tags_df

#initialize empty dataframes to append to
feeds_df = pd.DataFrame()
all_comments_df = pd.DataFrame()
all_tags_df = pd.DataFrame()
#go through all page ids
for page_id in page_ids:
    #pull facebook data
    feed_df,comments_df,tags_df = pull_data(page_id)
    #append to make larger dataset
    feeds_df = feeds_df.append(feed_df)
    all_comments_df = all_comments_df.append(comments_df)
    all_tags_df = all_tags_df.append(tags_df)

#write to excel
with pd.ExcelWriter('./nh19fbdata.xlsx') as writer:
    feeds_df.to_excel(writer, sheet_name='posts')
    all_comments_df.to_excel(writer, sheet_name='comments')
    all_tags_df.to_excel(writer, sheet_name='tags')

#write to sql
feeds_df.to_sql('posts',con=engine,if_exists="replace")
all_comments_df.to_sql('comments',con=engine,if_exists="replace")
all_tags_df.to_sql('tags',con=engine,if_exists="replace")
