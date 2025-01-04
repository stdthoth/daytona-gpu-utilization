FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3-pip
WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio transformers datasets accelerate torchmetrics

COPY *.py /workspace/
COPY train.txt /workspace/

CMD [ "bash" ]






