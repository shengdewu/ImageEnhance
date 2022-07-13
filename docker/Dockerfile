FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10

COPY codes /home/enhance/codes
COPY train_tool.py /home/enhance
COPY whl   /home/enhance/whl

RUN pip3 install einops && pip3 install /home/enhance/whl/*.whl

WORKDIR /home/enhance
ENTRYPOINT ["python3", "train_tool.py"]