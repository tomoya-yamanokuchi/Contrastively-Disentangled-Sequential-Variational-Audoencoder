Environments to be installed
-----------------------
```
- Ubuntu 20.04
- Python 3.8
- CUDA 11.4
- ROS noetic
- Pytorch 1.7.1
- MuJoCo 200 (I do not support 200 > )
- gym 
- mujoco-py 2.0.2
- Isaac Gym 1.0rc3
- Jupyter notebook
```



Preliminary  (If you are new to Docker)
-----------------------
1. Install Docker: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
2. Install nvidia-tool-kit: https://qiita.com/Hiroaki-K4/items/c1be8adba18b9f0b4cef





Main procedure
-----------------------
1. Place docker_ubuntu20_cudagl folder in /home/$USER/
2. In run.sh, change share folders (e.g., -v /home/workspace:) according to your environment. If you do not have these folders, just run "mkdir workspace" and "mkdir catkin_ws"

- In terminal, 
```
3. sh build.sh  (This takes about 20 min)
4. export DISPLAY=:0.0
5. export DISPLAY=:1.0
6. xhost local:root
7. sh run.sh
8. Let's start your docker life!! Run sample codes
  - python sample_code/pytorch_sample.py
  - python sample_code/gym_sample.py
  - python sample_code/mujoco_sample.py
  - cd isaacgym/python/examples/ && python 1080_balls_of_solitude.py
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
