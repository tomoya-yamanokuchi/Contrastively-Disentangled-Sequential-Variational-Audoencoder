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
$ python3.8 usecase/data_processing/data_split.py
```

It is recommended that this process should be divided into "train" and "test" and performed in order.
Because, it requires a lot of memory (RAM).


### run python script
```
(in docker container)
$ python3.8 usecase/train/train.py
```

The log file will be generated in the `./scripts/log` directory.



Procedure for evaluating the learned model
-----------------------

### set parameters in "conf/model/classifier_sprite.yaml"
```
ex.)
    group   : 'cdsvae_sprite'
    model   : '[c-dsvae]-[sprite_JunwenBi]-[dim_f=256]-[dim_z=32]-[500epoch]-[20230102135429]-cdsvae'
```
where, the `model` parameter is the directory name generated by the training.


### run test code
```
$ python3.8 usecase/test/test_sprite_cls.py
```

Then, you can check the evaluation results as follows:
```
Epoch[0/3] : [acc[%], IS, H_yx, H_y] = [89.06, 6.2501, 0.3391, 2.1717]
Epoch[1/3] : [acc[%], IS, H_yx, H_y] = [88.83, 6.3139, 0.3298, 2.1725]
Epoch[2/3] : [acc[%], IS, H_yx, H_y] = [88.87, 6.3336, 0.3274, 2.1732]
```


Image generation with the fixd component
-----------------------

### fixexd_motion:
![fixed_motion_left_0_1](https://user-images.githubusercontent.com/49630508/208916967-bd44c251-85f7-4903-a5be-52f27ea70f16.gif)
![fixed_motion_center_0_0](https://user-images.githubusercontent.com/49630508/208916960-f96450ae-2352-4635-ab5b-f814f75e7bd0.gif)
![fixed_motion_right_0_38](https://user-images.githubusercontent.com/49630508/208916970-c7135a38-4254-4315-a9b4-b834632b7483.gif)


### fixexd_content:
![fixed_content_green_hair_0_21](https://user-images.githubusercontent.com/49630508/208916886-ca07dc89-b251-4347-b99e-3cbf80858de3.gif)
![fixed_content_red_hair_0_0](https://user-images.githubusercontent.com/49630508/208916890-1d8ecf49-6692-4875-bdbe-2ccf165189ca.gif)
![fixed_content_white_hair_0_14](https://user-images.githubusercontent.com/49630508/208916893-e6f9717e-af88-4fd2-b13d-e8e0f14942e7.gif)