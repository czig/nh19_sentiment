#!/bin/bash

for i in {50..400..5}; do
    python ./topicModeling.py --type posts --date 2019-04-01 --num_topics 20 --num_passes 20 --iterations ${i} --ignore --logs
done
