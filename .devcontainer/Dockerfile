FROM osrf/ros:kilted-desktop

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  gdb \
  gdbserver \
  git \
  wget \
  curl \
  python3-pip \
  python3-venv \
  ros-dev-tools \
  python3-colcon-clean \
  libeigen3-dev \
  liblapacke-dev \
  libsuitesparse-dev \
  libopencv-dev \
  libboost-all-dev \
  libglfw3-dev \
  libepoxy-dev \
  libgl1-mesa-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev \
  mesa-utils \
  libgtest-dev \
  && rm -rf /var/lib/apt/lists/*

# Optional: build/install googletest libs if your projects need system-wide gtest linkage
RUN cd /usr/src/googletest && cmake -S . -B build && cmake --build build --target install
