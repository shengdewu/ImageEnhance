FROM dl.nvidia/cuda:10.2-cudnn7-devel-torch1.10

COPY ./ImageCureEnhance /code/cure

RUN rm -rf /code/cure/*.git && rm -rf /code/cure/*.idea && pip3 install einops

WORKDIR /code
ENTRYPOINT ["python3", "cure/train.py"]