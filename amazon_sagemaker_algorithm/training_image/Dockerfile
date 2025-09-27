FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

COPY /model /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV SAGEMAKER_PROGRAM training.py