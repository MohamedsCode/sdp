# Use the official lightweight Ubuntu 22.04 image
FROM ubuntu:22.04
# Set the working directory inside the container
WORKDIR /app
# Set environment variables to prevent interaction during
apt-get installs
ENV DEBIAN_FRONTEND=noninteractive
# Update the package list and install basic dependencies
(Python 3, git, cmake, etc.)
RUN apt-get update && apt-get install -y \
software-properties-common \
build-essential \
openjdk-11-jdk \
git \
ant \
wget \
curl \
cmake \
python3 \
python3-pip \
python3-venv \
python3-dev \
python3-tk \
python3-setuptools \
python3-wheel \
libfftw3-dev \
libeigen3-dev \sudo \
lsof \
iproute2 \
g++ \
python3-numpy \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*
# Clone the PX4 repository and install PX4 dependencies
RUN git clone --recursive https://github.com/PX4/PX4-
Autopilot.git /opt/PX4-Autopilot
WORKDIR /opt/PX4-Autopilot
RUN bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
# Clone the MAVSDK repository and build it
RUN git clone https://github.com/mavlink/MAVSDK.git
/opt/MAVSDK
WORKDIR /opt/MAVSDK
RUN cmake -DCMAKE_BUILD_TYPE=Release -
DBUILD_SHARED_LIBS=OFF -DBUILD_MAVSDK_SERVER=ON -
Bbuild/default -H.
RUN cmake --build build/default -j8
# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip
# Set up application dependencies from the requirements.txt
RUN apt-get remove --purge -y python3-blinker && apt-get
autoremove -y && apt-get clean
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
# Copy the rest of the application files into the container
COPY . /app/
# Expose the port for Flask (default port: 5000)
EXPOSE 5000
# Set the working directory back to /app and start the
application
WORKDIR /app
CMD ["python3", "app.py"]
