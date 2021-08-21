#!/bin/bash

if !(yum list installed jq >/dev/null 2>&1); then
    yum install -y jq
fi

model_type=${2:-$( jq -r '.strategy' $1 )}

if [ $model_type = federated ];then
    sh federated/run.sh $1
else
    python3 centralized_strategy.py --config_path $1
fi