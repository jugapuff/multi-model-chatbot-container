FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    transformers

RUN pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install grpc-wrapper==0.4 grpcio==1.14.1 six==1.11.0 googleapis-common-protos==1.5.3
RUN pip install kogpt2_transformers
COPY ./models /models
WORKDIR /models/
COPY main.py /models

CMD python3 main2.py
