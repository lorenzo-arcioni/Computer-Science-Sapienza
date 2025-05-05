FROM python:3.10

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Install buildozer dependencies
RUN apt update && apt install -y \
    git zip unzip openjdk-17-jdk \
    python3-pip ffmpeg libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev \
    libsdl2-ttf-dev libportmidi-dev libswscale-dev libavformat-dev \
    libavcodec-dev zlib1g-dev libjpeg-dev libfreetype6-dev \
    libgl1-mesa-dev libgles2-mesa-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libmtdev-dev xclip xsel \
    libffi-dev libssl-dev libpng-dev libusb-1.0-0 libusb-1.0-0-dev \
    libtool autoconf pkg-config \
    build-essential cmake && \
    pip install --upgrade pip

# Install buildozer and Cython
RUN pip install cython==0.29.36 buildozer==1.5.0

# Set workdir
WORKDIR /app
COPY . /app
