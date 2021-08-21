#!/bin/bash

function stop {
    echo "Stop service!"

    ps -ef | grep -E "fl" | grep -v grep | awk '{print $2}' | xargs kill -9
}

if !(yum list installed jq >/dev/null 2>&1); then
    yum install -y jq
fi

log_dir=${2:-$( jq -r '.federated.logdir' $1 )}

mkdir -p ${log_dir}
log_dir="${log_dir}/logs"
mkdir -p ${log_dir}

nb_clients=${2:-$(jq -r '.federated.number_of_clients' $1) }

python3  -m federated.fl_master.py --config_path $1 > ${log_dir}/master.log 2>&1 &
sleep 2
python3 -u -m federated.fl_scheduler.py --config_path $1 > ${log_dir}/scheduler.log 2>&1 &
sleep 5
python3 -u -m federated.fl_server.py --config_path $1 > ${log_dir}/server.log 2>&1 &
sleep 2

for ((i=0;i < ${nb_clients};i++))
do
    python3 -u -m federated.fl_trainer.py --id $i --config_path $1 > ${log_dir}/trainer$i.log 2>&1 &
    sleep 2
done

read -p 'running service ...'

trap stop EXIT