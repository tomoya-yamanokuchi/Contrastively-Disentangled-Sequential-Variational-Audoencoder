Overview of this repository
-----------------------
Pytorch-lightning implementation of [C-DSVAE](https://github.com/JunwenBai/C-DSVAE) with docker environment.




Main procedure for running the training code
-----------------------

## clone github repository
```
$ mkdir ~/workspace
$ cd ~/workspace
$ git clone https://github.com/tomoya-yamanokuchi/Contrastively-Disentangled-Sequential-Variational-Audoencoder.git
$ cd Contrastively-Disentangled-Sequential-Variational-Audoencoder
```

## set docker user-name in 'Dockerfile' and 'entrypoint.sh' to match your local host environment.
```
ex.) if your loacal host environment is 'tomoya-y@xxx:'
 (line 9 at Dockerfile) ENV UNAME user --> ENV UNAME tomoya-y
 (line 4 at entrypoint.sh) UNAME='user' --> UNAME='tomoya-y'
```

## build dokcer-image and run container
```
$ sh build.sh
$ sh run.sh
```

## run python script
```
(in docker container)
$ cd /home/$USER/workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder/
$
```