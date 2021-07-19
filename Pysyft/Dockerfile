FROM python:3.8.10

RUN git clone https://github.com/OpenMined/PySyft.git
RUN cd PySyft/packages/syft/docs && pip install -r requirements.txt

RUN pip install torch -U

RUN cd PySyft/packages/syft && python setup.py install

# RUN git clone https://github.com/OpenMined/PyGrid

# RUN pip install poetry

# RUN cd PyGrid/apps/domain && poetry install

RUN pip install mlxtend

# CMD cd PyGrid/apps/domain && ./run.sh --name bob --port 7000 --start_local_db