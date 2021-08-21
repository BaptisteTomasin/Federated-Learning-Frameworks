from paddle_fl.paddle_fl.core.scheduler.agent_master import FLScheduler
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)
                    
worker_num = params["federated"]["number_of_clients"]
server_num = 1
#Define number of worker/server and the port for scheduler
scheduler = FLScheduler(worker_num, server_num, port=params["federated"]["scheduler_port"])
scheduler.set_sample_worker_num(worker_num)
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
