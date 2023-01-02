#! /bin/bash

for i in `seq 5`
do
    docker run --rm -it --gpus all --privileged --net=host --ipc=host \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -v $HOME/.Xauthority:/home/$(id -un)/.Xauthority -e XAUTHORITY=/home/$(id -un)/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
    -v /dev/snd:/dev/snd -e AUDIODEV="hw:Device, 0" \
    -v /home/$USER/workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder/scripts:/home/$USER/workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder \
    -v /home/$USER/workspace/dataset:/home/$USER/workspace/dataset \
    docker_c_dsvae python3.8 usecase/train/train.py \
    model.loss.weight.kld_context=1.0 \
    model.loss.weight.kld_dynamics=1.0 \
    model.loss.weight.contrastive_loss_fx=0.0 \
    model.loss.weight.contrastive_loss_zx=0.0 \
    model.loss.weight.mutual_information_fz=0.0 \
    memo=$1
done