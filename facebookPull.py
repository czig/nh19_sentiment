import facebook, urllib3, requests
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///./nh19_fb.db')

page_ids = ['AFSOUTHNewHorizons','USEmbassyGeorgetown','southcom','dpiguyana','AFSouthern','NewsSourceGuyana','655452691211411','kaieteurnewsonline','demwaves','CapitolNewsGY','PrimeNewsGuyana','INews.Guyana','stabroeknews','NCNGuyanaNews','dailynewsguyana','actionnewsguyana','gychronicle']

with open('./access_token.txt','r') as token_file:
    access_token = token_file.read().replace('\n','')

graph = facebook.GraphAPI(access_token=access_token,version=3.1)

#function for getting reactions for object (superceded)
def get_all_reactions(object_id):
    reactions = graph.request(object_id + '?fields=reactions.type(LIKE).limit(0).summary(total_count).as(reactions_likes),reactions.type(LOVE).limit(0).summary(total_count).as(reactions_loves),reactions.type(WOW).limit(0).summary(total_count).as(reactions_wows),reactions.type(SAD).limit(0).summary(total_count).as(reactions_sads),reactions.type(ANGRY).limit(0).summary(total_count).as(reactions_angrys)')
    likes = reactions['reactions_likes']['summary']['total_count']
    loves = reactions['reactions_loves']['summary']['total_count']
    wows = reactions['reactions_wows']['summary']['total_count']
    sads = reactions['reactions_sads']['summary']['total_count']
    angrys = reactions['reactions_angrys']['summary']['total_count']
    return likes, loves, wows, sads, angrys

#subroutine to add reactions to object and abide by previous convention
def add_reactions(dict_object):
    dict_object['likes'] = dict_object['reactions_likes']['summary']['total_count'] 
    dict_object['loves'] = dict_object['reactions_loves']['summary']['total_count'] 
    dict_object['wows'] = dict_object['reactions_wows']['summary']['total_count'] 
    dict_object['sads'] = dict_object['reactions_sads']['summary']['total_count'] 
    dict_object['angrys'] = dict_object['reactions_angrys']['summary']['total_count'] 
    dict_object.pop('reactions_likes', None)
    dict_object.pop('reactions_loves', None)
    dict_object.pop('reactions_wows', None)
    dict_object.pop('reactions_sads', None)
    dict_object.pop('reactions_angrys', None)


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
    print(page_id)
    reactions_string = "reactions.type(LIKE).limit(0).summary(total_count).as(reactions_likes),reactions.type(LOVE).limit(0).summary(total_count).as(reactions_loves),reactions.type(WOW).limit(0).summary(total_count).as(reactions_wows),reactions.type(SAD).limit(0).summary(total_count).as(reactions_sads),reactions.type(ANGRY).limit(0).summary(total_count).as(reactions_angrys)"
    feed_gen = graph.get_all_connections(id=page_id, connection_name='feed',fields="id,created_time,message,message_tags,parent_id,shares," + reactions_string + ",comments{id,message,message_tags,created_time," + reactions_string + ",comment_count,comments{id,message,message_tags,created_time,comment_count," + reactions_string + "}}")
    feed = []
    all_comments = []
    all_message_tags = []
    #pull post data
    for post in feed_gen:
        if post[u'created_time'] > '2018-08-01':
            #handle information relating to post specifically (each post is a dict)
            print(post[u'created_time'])
            post[u'shares'] = post['shares']['count'] if 'shares' in post else 0
            post['page'] = page_id
            # add reactions to post dict and remove default nested dicts
            add_reactions(post)
            # handle comments on post (if any exist)
            if 'comments' in post.keys():
                #comments is a list of comments
                post['comments'] = post['comments']['data']
                for comment in post['comments']:
                    #keep track of parent and master page
                    comment[u'parent'] = comment[u'parent'][u'id'] if u'parent' in comment else ""
                    comment['parent_id'] = post[u'id']
                    comment['parent_type'] = 'post'
                    comment['page'] = page_id
                    # add reactions to comment dict and remove default nested dicts
                    add_reactions(comment)
                    #handle comments on comments (nested comments) (if any exist)
                    if 'comments' in comment.keys():
                        comment['comments'] = comment['comments']['data']
                        for nested_comment in comment['comments']:
                            # keep track of parent and page
                            nested_comment[u'parent'] = nested_comment[u'parent'][u'id'] if u'parent' in nested_comment else ""
                            nested_comment['parent_id'] = comment[u'id']
                            nested_comment['parent_type'] = 'comment'
                            nested_comment['page'] = page_id
                            # add reactions to nested comment dict and remove default nested dicts
                            add_reactions(nested_comment)
                            # handle message tags
                            handle_message_tags(nested_comment,'comment',all_message_tags)
                            # add nested comment to list 
                            all_comments.append(nested_comment)
                        #remove comments key/property once all nested comments have been appended to master list
                        comment.pop('comments', None)

                    # handle message tags
                    handle_message_tags(comment,'comment',all_message_tags)
                    # add comment to list 
                    all_comments.append(comment)
                #remove comments key/property once all comments have been appended to master list
                post.pop('comments', None)

            #handle message tags
            handle_message_tags(post,'post',all_message_tags)
            # add post to list 
            feed.append(post)

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
