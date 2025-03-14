FROM ubuntu:22.04

RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
  build-essential \
  cmake \
  gdb \
  libsuitesparse-dev \
  libeigen3-dev \
  libboost-all-dev \
  libopencv-dev \
  libglfw3-dev \
  git \
  libepoxy-dev \
  python3-pip \
  libgtest-dev
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git && cd Pangolin && mkdir build && cd build && cmake ../ && make -j4 && make install && ldconfig
