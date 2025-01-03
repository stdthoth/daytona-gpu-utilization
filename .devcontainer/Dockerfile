FROM pytorch/pytorch:2.4-cuda-12.1-cudnn9-runtime

WORKDIR /workspace

RUN pip install --upgrade pip
RUN pip install torchvision transformers datasets accelerate torchmetrics

COPY inference_workflow.py /workspace/inference_workflow.py
COPY finetune_workflow.py /workspace/finetune_workflow.py

CMD [ "python","finetune_workflow.py","inference_workflow" ]




