FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio transformers datasets accelerate torchmetrics

COPY *.py /workspace/
COPY data.txt /workspace/

CMD [ "bash" ]






