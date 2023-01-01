Overview of this repository
-----------------------
Pytorch-lightning implementation of [C-DSVAE](https://github.com/JunwenBai/C-DSVAE) with docker environment.


Main procedure for running the training code
-----------------------
```
$ mkdir ~/workspace
$ cd ~/workspace
$ git clone https://github.com/tomoya-yamanokuchi/Contrastively-Disentangled-Sequential-Variational-Audoencoder.git
$ cd Contrastively-Disentangled-Sequential-Variational-Audoencoder
$ sh build.sh
$ sh run.sh

--- docker container ---
$ cd /home/$USER/workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder/
$
```
Note:
- workspace is shared local folder to develop your codes.
- catkin_ws is shared local ROS folder.



If you get errors, check below
-----------------------
- Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock:
```
sudo usermod -aG docker ${USER}
su - ${USER}
```

- If you get "error", try remove
```
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
```
