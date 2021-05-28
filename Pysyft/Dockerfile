FROM python:3.8.10
ARG CUDA

RUN pip install --upgrade pip

RUN pip install syft

RUN if ["$CUDA" == "True"] ; then \
    pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html; \
  else \
    pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html; \
  fi

RUN pip install mlxtend