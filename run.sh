#! /bin/bash

# rm        : Automatically remove the container when it exits
# it        : -i + -t: For interactive processes (like a shell)
# gpus      : all, (or 0, 1, 2, ...)
# privileged: gives the container all the same capabilities as the host
# net=host  : share the same network name with host
# ipc=host  : share the same memory with host


docker run --rm -it --gpus all --privileged --net=host --ipc=host \
-e LOCAL_UID=$(id -u $USER) \
-e LOCAL_GID=$(id -g $USER) \
-v $HOME/.Xauthority:/home/$(id -un)/.Xauthority -e XAUTHORITY=/home/$(id -un)/.Xauthority \
-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
-v /dev/snd:/dev/snd -e AUDIODEV="hw:Device, 0" \
-v /home/$USER/workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder/scripts:/home/$USER/workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder \
-v /home/$USER/workspace/dataset:/home/$USER/workspace/dataset \
docker_c_dsvae bash