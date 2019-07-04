#!/bin/bash
TOPICS=(5 100)
CHUNKSIZE=(100 4000)
PASSES=(10 100)
UPDATE_EVERY=(0 1)
ITERATIONS=(50 1000)

for topic in "${TOPICS[@]}"; do
    for chunk in "${CHUNKSIZE[@]}"; do
        for pass in "${PASSES[@]}"; do
            for update in "${UPDATE_EVERY[@]}"; do
                for iteration in "${ITERATIONS[@]}"; do
                    python ./topicModeling.py --type posts --date 2019-04-01 --num_topics $topic --chunk_size $chunk --num_passes $pass --update_every $update --iterations $iteration --name doe1 --ignore --logs
                done
            done
        done
    done
done
