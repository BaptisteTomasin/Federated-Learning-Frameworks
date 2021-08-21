#!/bin/bash

if !(yum list installed jq >/dev/null 2>&1); then
    apt install -y jq
fi

strategy=${2:-$( jq -r '.strategy' $1 )}

if [ $strategy = federated ];then
    python3 -m federated.federated_strategy.py --config_path $1
else
    python3 centralized_strategy.py --config_path $1
fi