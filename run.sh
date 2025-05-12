#!/bin/bash

cd "/home/monoshi/Analyzing-LLM-Performance-on-Different-Code-Complexity/LLM Utils/"
CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

for model_id in {7..12}; do
    echo "Clearing cache..."
    rm -rf $CACHE_PATH
    
    echo "Running model_id $model_id..."
    python main.py --data_id 0 --model_id "$model_id" --pt_id 0 --language python --prediction output
    python main.py --data_id 0 --model_id "$model_id" --pt_id 0 --language python --prediction input
    sleep 1
    
    echo "Completed model_id $model_id"
    echo "----------------------------------"
done

echo "All model runs completed!"