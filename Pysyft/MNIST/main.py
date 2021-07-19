# stdlib
import base64
import json

# third party
import jwt
import requests
import torch as th
from websocket import create_connection

# syft absolute
import syft as sy
from syft import deserialize
from syft import serialize
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.lib.python.int import Int
from syft.lib.python.list import List
from syft.proto.core.plan.plan_pb2 import Plan as PlanPB
from syft.proto.lib.python.list_pb2 import List as ListPB

th.random.manual_seed(42)

# Define the model
####################

class MLP(sy.Module):
    def __init__(self, torch_ref):
        super().__init__(torch_ref=torch_ref)
        self.l1 = self.torch_ref.nn.Linear(784, 100)
        self.a1 = self.torch_ref.nn.ReLU()
        self.l2 = self.torch_ref.nn.Linear(100, 10)

    def forward(self, x):
        x_reshaped = x.view(-1, 28 * 28)
        l1_out = self.a1(self.l1(x_reshaped))
        l2_out = self.l2(l1_out)
        return l2_out

# Define Training Plan
#######################

def set_params(model, params):
    for p, p_new in zip(model.parameters(), params):
        p.data = p_new.data


def cross_entropy_loss(logits, targets, batch_size):
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    return -(targets * log_probs).sum() / batch_size


def sgd_step(model, lr=0.1):
    with ROOT_CLIENT.torch.no_grad():
        for p in model.parameters():
            p.data = p.data - lr * p.grad
            p.grad = th.zeros_like(p.grad.get())

local_model = MLP(th)

# Local Train
###############
@make_plan
def train(
    xs=th.rand([64 * 3, 1, 28, 28]),
    ys=th.randint(0, 10, [64 * 3, 10]),
    params=List(local_model.parameters()),
):

    model = local_model.send(ROOT_CLIENT)
    set_params(model, params)
    for i in range(1):
        indices = th.tensor(range(64 * i, 64 * (i + 1)))
        x, y = xs.index_select(0, indices), ys.index_select(0, indices)
        out = model(x)
        loss = cross_entropy_loss(out, y, 64)
        loss.backward()
        sgd_step(model)

    return model.parameters()

# Define Averaging Plan
#########################

@make_plan
def avg_plan(
    avg=List(local_model.parameters()), item=List(local_model.parameters()), num=Int(0)
):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

# Config & keys
################

# Config

name = "mnist"
version = "1.0"

client_config = {
    "name": name,
    "version": version,
    "batch_size": 64,
    "lr": 0.1,
    "max_updates": 3,  # custom syft.js option that limits number of training loops per worker
}

server_config = {
    "min_workers": 2,
    "max_workers": 2,
    "pool_selection": "random",
    "do_not_reuse_workers_until_cycle": 6,
    "cycle_length": 28800,  # max cycle length in seconds
    "num_cycles": 30,  # max number of cycles
    "max_diffs": 1,  # number of diffs to collect before avg
    "minimum_upload_speed": 0,
    "minimum_download_speed": 0,
    "iterative_plan": True,  # tells PyGrid that avg plan is executed per diff
}

# Keys

def read_file(fname):
    with open(fname, "r") as f:
        return f.read()

private_key = read_file("example_rsa").strip()
public_key = read_file("example_rsa.pub").strip()

server_config["authentication"] = {
    "type": "jwt",
    "pub_key": public_key,
}

# Host in PyGrid
##################

grid_address = "localhost:7000"

grid = ModelCentricFLClient(address=grid_address, secure=False)
# grid.connect()

# response = grid.host_federated_training(
#     model=local_model,
#     client_plans={"training_plan": train},
#     client_protocols={},
#     server_averaging_plan=avg_plan,
#     client_config=client_config,
#     server_config=server_config,
# )

# print(response)

# # Authenticate for cycle
# ###########################

# # Helper function to make WS requests
# def sendWsMessage(data):
#     ws = create_connection("ws://" + grid_address)
#     ws.send(json.dumps(data))
#     message = ws.recv()
#     return json.loads(message)

# auth_token = jwt.encode({}, private_key, algorithm="RS256").decode("ascii")

# auth_request = {
#     "type": "model-centric/authenticate",
#     "data": {
#         "model_name": name,
#         "model_version": version,
#         "auth_token": auth_token,
#     },
# }
# auth_response = sendWsMessage(auth_request)
# print(auth_response)

# # Do cycle request
# #####################
# cycle_request = {
#     "type": "model-centric/cycle-request",
#     "data": {
#         "worker_id": auth_response["data"]["worker_id"],
#         "model": name,
#         "version": version,
#         "ping": 1,
#         "download": 10000,
#         "upload": 10000,
#     },
# }
# cycle_response = sendWsMessage(cycle_request)
# print("Cycle response:", json.dumps(cycle_response, indent=2).replace("\\n", "\n"))

# # Download model
# ###################

# worker_id = auth_response["data"]["worker_id"]
# request_key = cycle_response["data"]["request_key"]
# model_id = cycle_response["data"]["model_id"]
# training_plan_id = cycle_response["data"]["plans"]["training_plan"]

# def get_model(grid_address, worker_id, request_key, model_id):
#     req = requests.get(
#         f"http://{grid_address}/model-centric/get-model?worker_id={worker_id}&request_key={request_key}&model_id={model_id}"
#     )
#     model_data = req.content
#     pb = ListPB()
#     pb.ParseFromString(req.content)
#     return deserialize(pb)

# # Model
# model_params_downloaded = get_model(grid_address, worker_id, request_key, model_id)
# print("Params shapes:", [p.shape for p in model_params_downloaded])

# print(model_params_downloaded[0])

# # Download & Execute Plan

# req = requests.get(
#     f"http://{grid_address}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list"
# )
# pb = PlanPB()
# pb.ParseFromString(req.content)
# plan = deserialize(pb)

# xs = th.rand([64 * 3, 1, 28, 28])
# ys = th.randint(0, 10, [64 * 3, 10])

# (res,) = plan(xs=xs, ys=ys, params=model_params_downloaded)

# # Report Model diff

# diff = [orig - new for orig, new in zip(res, local_model.parameters())]
# diff_serialized = serialize((List(diff))).SerializeToString()

# params = {
#     "type": "model-centric/report",
#     "data": {
#         "worker_id": worker_id,
#         "request_key": request_key,
#         "diff": base64.b64encode(diff_serialized).decode("ascii"),
#     },
# }

# sendWsMessage(params)

# # Check new model

# req_params = {
#     "name": name,
#     "version": version,
#     "checkpoint": "latest",
# }

# res = requests.get(f"http://{grid_address}/model-centric/retrieve-model", req_params)

# params_pb = ListPB()
# params_pb.ParseFromString(res.content)
# new_model_params = deserialize(params_pb)

# print(new_model_params[0])