FROM nvidia/cudagl:11.1.1-base-ubuntu18.04

# Install packages without prompting the user to answer any questions
ENV DEBIAN_FRONTEND=noninteractive

#####################################################
# switch from root to user
#####################################################
ENV UNAME user
RUN useradd -m $UNAME
WORKDIR /home/$UNAME
# For uid, gid
RUN apt-get update -qq && apt-get -y install gosu
COPY  assets/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


#####################################################
# Install common apt packages
#####################################################
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get install -y \
	### utility
	locales \
	xterm \
	dbus-x11 \
	terminator \
	sudo \
	### tools
	unzip \
	lsb-release \
	curl \
	ffmpeg \
	net-tools \
	software-properties-common \
	subversion \
	libssl-dev \
	### Development tools
	build-essential \
	htop \
	git \
	vim \
	gedit \
	gdb \
	valgrind \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*


#####################################################
# Set locale & time zone
#####################################################
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ=Asia/Tokyo


# #####################################################
# # Python 3.8
# #####################################################
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
  python3.8 \
  python3-pip \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*
RUN echo "alias python='python3.8'" >> /root/.bashrc

RUN apt-get update && \
	apt-get install -y wget && \
	wget https://bootstrap.pypa.io/get-pip.py && \
	python3.8 get-pip.py


#####################################################
# Install common pip packages
#####################################################
RUN apt-get update
COPY pip/requirements.txt requirements.txt
RUN python3.8 -m pip install -r requirements.txt


#####################################################
# Pytorch Lightning
#####################################################
COPY pip/requirements_pytorch_lightning.txt requirements_pytorch_lightning.txt
RUN python3.8 -m pip install -r requirements_pytorch_lightning.txt