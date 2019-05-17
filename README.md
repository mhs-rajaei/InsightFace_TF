## Insight Face in TensorFlow
### My Changes
#### Thanks [Chen Wei](https://github.com/auroua/InsightFace_TF), [David Sandberg](https://github.com/davidsandberg/facenet), [Victor Zhang](https://github.com/VictorZhang2014/facenet) and many other peoples.
##### Removing of tfrecords and adding parser for loading raw 'jpeg' or 'png' images.
##### Custom validation dataset and custom 'pairs.txt' creation from [here](https://github.com/VictorZhang2014/facenet/blob/master/mydata/generate_pairs.py) (Some bugs removed).
##### Ready for use in google colab (you must upload your code, model, etc. Use this link for help: [here](https://zerowithdot.com/colab-workspace).
##### ArgParse removed and Class Args added.

### Restoring ckpt (Pre-trained model) and validate this model on the custom dataset
* The custom dataset must have this structure (don't worry about empty folders or other files in the custom dataset. Name of folder or images doesn't matter, but make sure that your name doesn't contain tab or '\t', I use tab as the delimiter in 'pairs.txt':

```
The custom daataset:
    ├── folder1
    │   ├── image1.ext
    │   ├── image2.ext
    │   ├── image3.ext
    │   ├── ...
    ├── folder2
    │   ├── image1.ext
    │   ├── image2.ext
    │   ├── image3.ext
    │   ├── image4.ext
    │   └── ...
    ├── folder3
    │   ├── image1.ext
    │   ├── image2.ext
    │   ├── ...
    ...
    ...
```
* Download pre-trained weights of model d, you can download this model from  [google drive](https://drive.google.com/open?id=19PbuQP2wDn-vXfNfc4HFPSE1rZNG18VG).
* In the following and in each step, add the correct path to class Args.
* Align your custom dataset with [align_dataset_mtcnn.py](https://github.com/mhs-rajaei/InsightFace_TF/blob/master/align/align_dataset_mtcnn.py).
* Generate 'pairs.txt' with ['generate_pairs.py'](https://github.com/mhs-rajaei/InsightFace_TF/blob/master/generate_pairs.py) (set image 
extension to 'img_ext' in 'generate_pairs.py')
* Use ['eval_ckpt_file.py'](https://github.com/mhs-rajaei/InsightFace_TF/blob/master/eval_ckpt_file.py).


#### Tasks
* ~~mxnet dataset to tfrecords~~
* ~~backbone network architectures [vgg16, vgg19, resnet]~~
* ~~backbone network architectures [resnet-se, resnext]~~
* ~~LResNet50E-IR~~
* ~~LResNet100E-IR~~
* ~~Additive Angular Margin Loss~~
* ~~CosineFace Loss~~
* ~~train network code~~
* ~~add validate during training~~
* ~~multi-gpu training~~
* ~~combine losses~~ contributed by RogerLo.
* evaluate code


#### Training Tips(Continual updates)
* If you can't use large batch size(>128), you should use small learning rate
* If you can't use large batch size(>128), you can try batch renormalization(file `L_Resnet_E_IR_RBN.py`)
* If use multiple gpus, you should keep at least 16 images each gpu.
* Try [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf), you can use the code `L_Resnet_E_IR_GBN.py`
* Using the current model, and the lr schedule in `train_nets.py`, you can get the results as `model c`
* The bug about model size is 1.6G have fixed based on issues #9. If you want to get a small model, you should use `L_Resnet_E_IR_fix_issues9.py`
* multi-gpu training code's bug have fixed. If you want to use the correct version, you should use `train_nets_mgpu_new.py`


#### Training models (Continual updates)

##### model A
| model name    | depth| normalization layer |batch size| total_steps | download | password |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model A | 50 |group normalization|16| 1060k |[model a](https://pan.baidu.com/s/1qWrDCTFlQXlFcBR-dqR-6A)|2q72|

###### accuracy
| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9897|
| cfp_ff |0.9876|
| cfp_fp |0.84357|
| age_db30 |0.914|


##### model B
| model name    | depth| normalization layer |batch size| total_steps| download | password |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model B | 50 |batch normalization|16| 1100k |[model_b](https://pan.baidu.com/s/11KDqOkF4ThO7mnQQaNO9bA) |h6ai|

###### accuracy
| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9933|
| cfp_ff |0.99357|
| cfp_fp |0.8766|
| age_db30 |0.9342|



##### model C
| model name    | depth| normalization layer |batch size| total_steps| download | password |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model C | 50 |batch normalization|16| 1950k |[model_c](https://pan.baidu.com/s/1ZlDcQPBh0znduSH6vQ_Q8Q) |8mdi|

###### accuracy
| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9963|
| cfp_ff |0.99586|
| cfp_fp |0.9087|
| age_db30 |0.96367|


##### model D
| model name    | depth| normalization layer |batch size| total_steps| model_size| download | password |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model D | 50 |batch normalization|136| 710k | 348.9MB |[model_d](https://pan.baidu.com/s/1tQYMqfbG36wg1cXKHVoMVw) |amdt|

###### accuracy
| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9968|
| cfp_ff |0.9973|
| cfp_fp |0.9271|
| age_db30 |0.9725|



#### Requirements
1. TensorFlow >= 1.7 and <= 1.13.1
2. TensorLayer = 1.7
3. cuda8&cudnn6 or cuda9&cudnn7 (if you want use tensorflow gpu)
4. Python3


#### Max Batch Size Test
###### Environment

| GPU    | cuda| cudnn | TensorFlow |TensorLayer|Maxnet |Gluon|
| ----- |:-----:|:-----:|:------:|:---:|:------:|:---:|
| Titan xp | 9.0 |7.0|1.6|1.7 |1.1.0|1.1.0 |

###### Results

| DL Tools        | Max BatchSize(without bn and prelu)| Max BatchSize(with bn only) | Max BatchSize(with prelu only) |Max BatchSize(with bn and prelu)|
| ------------- |:-------------:|:--------------:|:------------:|:------------:|
| TensorLayer      | (8000, 9000) |(5000, 6000)|(3000, 4000)|(2000, 3000) |
| Mxnet      | (40000, 50000) |(20000, 30000)|(20000, 30000)|(10000, 20000) |
| Gluon      | (7000, 8000) |(3000, 4000)|no official method| None |

> (8000, 9000) : 8000 without OOM, 9000 OOM Error

###### Test Code

|TensorLayer| Maxnet | Gluon |
| ----- |:-----:|:-----:|
| [tensorlayer_batchsize_test.py](https://github.com/auroua/InsightFace_TF/blob/master/test/benchmark/tensorlayer_batchsize_test.py) | [mxnet_batchsize_test.py](https://github.com/auroua/InsightFace_TF/blob/master/test/benchmark/mxnet_batchsize_test.py) |[gluon_batchsize_test.py](https://github.com/auroua/InsightFace_TF/blob/master/test/benchmark/gluon_batchsize_test.py)|



#### pretrained model download link
* [resnet_v1_50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
* [resnet_v1_101](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)
* [resnet_v1_152](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)
* [vgg16](http://www.cs.toronto.edu/~frossard/post/vgg16/)
* [vgg19](https://github.com/machrisaa/tensorflow-vgg)


#### References
1. [InsightFace mxnet](https://github.com/deepinsight/insightface)
2. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
3. [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)
3. [tensorlayer_vgg16](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_vgg16.py)
4. [tensorlayer_vgg19](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_vgg19.py)
5. [tf_slim](https://github.com/tensorflow/models/tree/master/research/slim)
6. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
7. [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
8. [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)