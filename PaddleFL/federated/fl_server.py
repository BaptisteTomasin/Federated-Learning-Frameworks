from paddle_fl.paddle_fl.core.server.fl_server import FLServer
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help = "path to the config file")
args = parser.parse_args()

with open(args.config_path, 'r') as fp: 
    params = json.load(fp)

server = FLServer()
server_id = 0
job_path = params["federated"]["job_path"]
print("job_path: ",job_path)
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
job._scheduler_ep = "127.0.0.1:"+str(params["federated"]["scheduler_port"])
print("IP address for scheduler: ",job._scheduler_ep)
server.set_server_job(job)
server._current_ep = "127.0.0.1:"+str(params["federated"]["server_port"])
print("IP address for server: ",server._current_ep)
server.start()