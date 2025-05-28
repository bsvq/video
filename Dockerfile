ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
#ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM $BASE_IMAGE

ENV     DEBIAN_FRONTEND=noninteractive
ENV     NVIDIA_VISIBLE_DEVICES=all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV     NVIDIA_REQUIRE_CUDA=cuda>=11.4
ENV     NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,compat32
ENV     HOME=/home
ENV     HOME_ENV=/home/venv

RUN     apt-get update && apt-get install -y tzdata sudo

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    wget \
    bash \
    cmake \
    curl \
    git \
    vim
RUN apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install libgl1 -y && apt-get install curl -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# RUN apt install -y python3-dev

# RUN python3 --version
# RUN apt-get install -y python3-pip
# RUN pip --version
RUN cd $HOME
# CMD ["/bin/bash"]

RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv
RUN apt install -y python3.11-distutils && wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py
RUN apt-get install -y python3-pyaudio portaudio19-dev 
RUN python3.11 -m venv $HOME_ENV
ENV PATH="$HOME_ENV/bin:$PATH"
# RUN source $HOME_ENV/bin/activate
RUN which python3.11

#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYTHONUNBUFFERED=1

#ENV CONDA_DIR=/home/miniconda3
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#RUN sh Miniconda3-latest-Linux-x86_64.sh -b -u -p $CONDA_DIR
#ENV PATH=$CONDA_DIR/bin:$CONDA_DIR/Scripts:$PATH

#RUN conda init
#RUN conda --version
#RUN conda create -n nerfstream python==3.10 pip
#RUN conda env list
#RUN activate nerfstream

#RUN conda install pytorch torchvision cudatoolkit -c pytorch
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install wheel
#RUN pip install TTS
#RUN pip install git+https://github.com/coqui-ai/TTS
COPY requirements.txt ./
# CMD ["/bin/bash"]
RUN pip3 install -r requirements.txt
# RUN apt-get update && apt-get install -y python3-opencv
# RUN pip3 install opencv-python
# RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
# RUN pip3 install 'tensorflow[and-cuda]'

# RUN pip3 install ffmpeg-python

COPY . /home/nerfstream
WORKDIR /home/nerfstream

#EXPOSE 1-65535/udp
EXPOSE 10000-10200/udp
EXPOSE 8089
# CMD ["/bin/bash"]
# ENTRYPOINT ["python", "app.py"]
# ARG AVATAR_ID
# ENV AVATAR_ID=$AVATAR_ID
CMD ["/bin/bash", "-c", "python3 app.py --avatar_id ${AVATAR_ID} --fps ${AVATAR_FPS}"]
# CMD ["--avatar_id avatar_blonde"]
