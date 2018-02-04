FROM amd64/debian:stretch-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-dev \
	python3-pip \
	python3-setuptools

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        nltk

RUN pip3 --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.0-cp35-cp35m-linux_x86_64.whl

RUN mkdir /tf
WORKDIR "/tf"

CMD ["bash"]