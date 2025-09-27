FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

COPY inference.py /opt/conda/lib/python3.10/site-packages/sagemaker_pytorch_serving_container/default_pytorch_inference_handler.py

COPY modules.py /opt/conda/lib/python3.10/site-packages/sagemaker_pytorch_serving_container/modules.py