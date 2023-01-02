Overview of this repository
-----------------------
Pytorch-lightning implementation of [C-DSVAE](https://github.com/JunwenBai/C-DSVAE) with docker environment.




Main procedure for running the training code
-----------------------

### clone github repository
```
(in local host)

$ mkdir ~/workspace
$ cd ~/workspace
$ git clone https://github.com/tomoya-yamanokuchi/Contrastively-Disentangled-Sequential-Variational-Audoencoder.git
$ cd Contrastively-Disentangled-Sequential-Variational-Audoencoder
```

### set docker user-name in "Dockerfile" and "entrypoint.sh" to match your local host environment.
```
ex.) if your loacal host environment is "tomoya-y@xxx:"
 (line 9 at Dockerfile) ENV UNAME user --> ENV UNAME tomoya-y
 (line 4 at entrypoint.sh) UNAME='user' --> UNAME='tomoya-y'
```

### build dokcer-image and run container
```
(in local host)

$ sh build.sh
```

### download [Sprite dataset](https://drive.google.com/file/d/1PLaEmvn7xrA_rNPCUnWYJd-YQ3HW0EDo/view)
```
(in local host)

$ mkdir -p ~/workspace/dataset/Sprite
$ cd ~/workspace/dataset/Sprite
$ (download zip format dataset from above link and extract)
```

### Split overall dataset into "train.pkl" and "test.pkl"
```
(in local host)
$ sh run.sh

(in docker container)
$ cd workspace/Contrastively-Disentangled-Sequential-Variational-Audoencoder/
$ python3.8 usecase/data_processing/data_split.py
```

It is recommended that this process should be divided into "train" and "test" and performed in order.
Because, it requires a lot of memory (RAM).


### run python script
```
(in docker container)
$ python3.8 usecase/train/train.py
```

The log file will be generated in the `. /log` directory.