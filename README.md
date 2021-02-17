# YOLOv4_tensorflow | [中文说明](README.cn.md)
* Implement yolov4 with tensorflow1
* tf.data.pipline
</br>

* rdc01234@163.com

## introductions
* run the following command.
```
python val_voc.py
```
* if have no error, it's ok

## converts yolov4.weights to fitting our code
* refer to [this weights convert file](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), I'm already converted yolov4.weights to this project.
* you can download yolov4.weights from [baidu cloud](https://pan.baidu.com/s/1VnX5lWT4CkHyqq0JQSllmA)  Extraction code: wm1j
* **puting yolov4.weights into the "yolo_weights" folder, and run the command.**
```
python convert_weight.py
python test_yolo_weights.py
```
* the ckpt weights file wound exits in the 'yolo_weights' folder(exists in the baidu cloud)
* and you'll see some images like this, it seems perfect
* 
![image](coco_save/dog.jpg)
* 
![image](coco_save/eagle.jpg)
* 
![image](coco_save/person.jpg)
* 
* the weights_name.txt contains all layer's name of the network 

## training on VOC2007 and VOC2012
* open config.py and modify the **voc_root_dir** to the root path of your VOC dataset </br>
* just like this
```
path_to_voc_root_dir
        voc_dir_ls[0] (VOC_2007)
        |       |Annotations
        |       |JPEGImages
        |       |...
        voc_dir_ls[1] (VOC_2012)
        |        |Annotations
        |        |JPEGImages
        |        |...
        others_folder_name
```
* run the command
```
python train_voc.py
```
* put test images into **voc_test_pic** folder, and run the following command after the model training finished.</br>
```
python val_voc.py
```
* it's the result of our code training(input_size:416*416, batch_size:2, lr:2e-4, optimizer:momentum) for a day(364999 steps), not bad
* 
![image](voc_save/000302.jpg)
* 
![image](voc_save/000288.jpg)
* 

* **all configuration parameters are in the config.py, you can modify them according to your actual situation**
* it's the image of loss value, and seems that the lr is too lower(2e-4), we should set it larger.
```
python show_loss.py 20 300
```
* 
![image](loss.png)

## training on VOC2007 and VOC2012 with tf.data pipline
* just same with **training on VOC2007 and VOC2012** but run training file
```
train_voc_tf_data.py
```

## training on own dataset
* **On January 5, 2021, the related code of training own dataset was deleted, and if you want to training your own dataset, following **src/Data_voc.py**  and just modify the **__init_args** function, it will be working great**
* 

## converts ckpt model to pb model
* open ckpt2pb.py , and modify 'ckpt_file_dir', 'class_num', 'anchors'. run.
```
python ckpt2pb.py
```
* you will see a pb model named 'model.pb' in 'ckpt_file_dir'
* **if you want to use pb model, you will learn more from 'val_pb.py'**
* run python file
```
python val_pb.py
```
* you will see the detection results with pb model 
* you can download pb model from [baidu cloud](https://pan.baidu.com/s/1VnX5lWT4CkHyqq0JQSllmA)  Extraction code: wm1j

## some tips about config.py and training the model
1. the parameters of **width and height** in config.py should be 608, but i have not a powerful GPU, that is why i set them as 416
2. actually, you can replace mish activation with leaky_relu to save GPU memory, and then you can set batch_size from 2 to 4
3. learning rate do not set too large
4. when the loss value is Nan, please lower your learning rate.

## my device
GPU : 1660ti (ASUS) 6G</br>
CPU : i5 9400f</br>
mem : 16GB</br>
os  : ubuntu 18.04</br>
cuda: 10.2</br>
cudnn : 7</br>
python : 3.6.9</br>
tensorflow-gpu:1.14.0</br>
numpy : 1.18.1</br>
opencv-python : 4.1.2.30</br>
