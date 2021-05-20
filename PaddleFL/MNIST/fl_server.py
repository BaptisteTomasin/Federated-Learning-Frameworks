from paddle_fl.paddle_fl.core.server.fl_server import FLServer
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob

server = FLServer()
server_id = 0
job_path = "fl_job_config"
print("job_path: ",job_path)
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
job._scheduler_ep = "127.0.0.1:9091"  # IP address of scheduler
print("IP address for scheduler: ",job._scheduler_ep)
server.set_server_job(job)
server._current_ep = "127.0.0.1:8181"  # IP address of server
print("IP address for server: ",server._current_ep)
server.start()