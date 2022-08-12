FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10.0

COPY ./codes /home/enhance/codes
COPY ./train_tool.py /home/enhance
COPY ./whl   /home/enhance/whl

RUN pip3 install einops && pip3 install /home/enhance/whl/*.whl

RUN cd /home/enhance/codes/network/lut/trilinear/cpp && pwd && rm -rf build && rm -rf trilinear.* && sh setup.torch1.x.sh

WORKDIR /home/enhance

ENTRYPOINT ["python3", "train_tool.py"]