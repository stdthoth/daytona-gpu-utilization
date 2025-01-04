FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio transformers datasets accelerate torchmetrics

COPY *.py /workspace/
COPY train.txt /workspace/

CMD [ "bash" ]






