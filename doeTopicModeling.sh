#!/bin/bash
#TOPICS=(5 100)
#CHUNKSIZE=(100 4000)
#PASSES=(10 100)
#UPDATE_EVERY=(0 1)

#for topic in "${TOPICS[@]}"; do
    #for chunk in "${CHUNKSIZE[@]}"; do
        #for pass in "${PASSES[@]}"; do
            #for update in "${UPDATE_EVERY[@]}"; do
            #done
        #done
    #done
#done

#hardcode since for loop would run too many
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 5 --chunk_size 2050 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 100 --chunk_size 2050 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 100 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 4000 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 10 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 100 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 55 --update_every 1 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 55 --update_every 0 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 55 --update_every 3001 --iterations 50 --name doe2 --ignore --logs
