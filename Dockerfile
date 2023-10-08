FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
        build-essential \
        curl \
        git-core \
        iputils-ping \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        wget \
        gcc


RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install --yes nodejs

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY package.json /usr/src/app/
RUN npm install
COPY . /usr/src/app

CMD ["npm", "start:dev"]
