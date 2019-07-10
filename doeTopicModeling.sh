#!/bin/bash
TOPICS=(29 77)
CHUNKSIZE=(2050 4000)
UPDATE_EVERY=(0 3000)

for topic in "${TOPICS[@]}"; do
    for chunk in "${CHUNKSIZE[@]}"; do
        for update in "${UPDATE_EVERY[@]}"; do
            python ./topicModeling.py --type posts --date 2019-04-01 --num_topics ${topic} --chunk_size ${chunk} --num_passes 10 --update_every ${update} --iterations 50 --name doe3 --ignore --logs
        done
    done
done

#hardcode since for loop would run too many
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 29 --chunk_size 3025 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 77 --chunk_size 3025 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 3025 --num_passes 10 --update_every 3000 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 3025 --num_passes 10 --update_every 0 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 2050 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 4000 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 3025 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 3025 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 53 --chunk_size 3025 --num_passes 10 --update_every 4500 --iterations 50 --name doe3 --ignore --logs
