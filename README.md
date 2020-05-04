# YOLOv4_tensorflow
yolov4的tensorflow实现. <br/>
Implement yolov4 with tensorflow<br/>
持续更新<br/>
continuous update the code<br/>
同志们，五一快乐。</br>
hey guys, happy International Workers ' Day.</br>
</br>
终于解决了 ciou_loss NAN的问题</br>
finally, the problem of CIOU_LOSS value equal to NAN was solved </br>
</br>

## 二战中科院计算所失败，求老师调剂收留
## I want to be a graduate student.
277118506@qq.com<br/>
rdc01234@163.com<br/>

## 使用说明
## introductions
执行命令. <br/>
run the following command.
```
python val.py
```
如果没有报错, 就没问题<br/>
if have no error, it's ok

### 转换 yolov4.weights (没有成功)
### convert yolov4.weights to fit our code(i failed )
参考[这个权重转换文件](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), 我将 yolov4.weights 转换到了自己的代码中
执行命令<br/>
refer to [this weights convert file](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), i converted yolov4.weights to this project.<br/>
<br/>
将下载好的 yolov4.weights 放到 yolo_weights 文件夹下, 执行命令<br/>
put yolov4.weights into the "yolo_weights" folder, and run the command.
```
python convert_weight.py
python test_yolo_weights.py
```
会在 yolo_weights 文件夹下生成权重文件<br/>
the ckpt weights file wound exits in the 'yolo_weights' folder<br/>
<br/>
权重转换是能够转换过来, 但是并不能跑出效果, 我确定全部权重都存放到了自己的项目中，因为权重的数量都是 64429405 个，排查了一天还是不知道哪里不对,我猜测是卷积层加载数据的顺序错了, 希望有同志能够帮忙解决这个问题, 谢谢了。<br/>
i'm sure that the weights file convertion to the ckpt is successful, because the number of both weights are 64429405, but maybe the order of conv layer in this code is different with the yolov4, i worked for this for a day, still have no idea about it, I hope comrades can give me some help, thanks.<br/>
<br/>
weights_name.txt 文件中存放的是图模型的卷积层和bn的名字<br/>
the weights_name.txt contains all model layer's name of the network <br/>
<br/>
数据增强策略没有实现，以后会慢慢更新</br>
data enhancement strategies have not been implemented, and will be updated slowly in the future.</br>
<br/>

### 训练自己的数据集
### train with own dataset
./data/JPEGImages 文件夹中存放用labelme标注json文件的jpg图片和对应的json文件, 参考我给的文件夹<br/>
The jpg image and the corresponding json file which marked with 'labelme' are stored in the folder "./data/JPEGImages", just like what I do<br/>
<br/>
然后在 ./data 文件夹下执行 python 命令, 会自动产生 label 文件和 train.txt 文件<br/>
and then, go to the folder "./data", execute the following python command, it automatically generates label files and train.txt
```
python generate_labels.py
```
继续执行命令,得到 anchor box<br/>
excute the python command, to get anchor box
```
python k_means.py
```
<br/>
打开 config.py, 将得到的 anchor box 写入到第六行，就像这样<br/>
anchors = 12,19, 19,27, 18,37, 21,38, 23,38, 26,39, 31,38, 39,44, 67,96<br/>
open config.py, write the anchor box to line 6, just like this<br/>
<br/>
所有的配置参数都在 config.py 中，你可以按照自己的实际情况来修改<br/>
all configuration parameters are in the config.py, you can modify them according to your actual situation<br/>
<br/>
配置完成,执行命令<br/>
ok, that's all, execute the command

```
python train.py
```
<br/>
训练完成后,将测试图片放到 test_pic 文件夹下,执行命令<br/>
put test images into test_pic folder, and run the following command after the model training.<br/>

```
python val.py
```
<br/>
这是我用54张图片训练了 2000 步(十分钟)的结果，效果还不错<br/>
this image is the result of training 2000 steps (10 minutes) with 54 pictures, it looks not bad. <br/>
![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/save/60.jpg)

### 有关 config.py 和训练的提示
### some tips with config.py and train the model
config.py 中的 width 和 height 应该是 608，显存不够才调整为 416 的<br/>
the parameters of width and height in config.py should be 608, but i have not a powerful GPU, that is why i set them as 416<br/>
<br/>
学习率不宜设置太高<br/>
learning rate do not set too large<br/>
<br/>
如果出现NAN的情况，请降低学习率</br>
when the loss value is Nan, please lower your learning rate.
</br>

### 自己的设备
### my device
GPU : 1660ti (华硕猛禽) 6G<br/>
CPU : i5 9400f<br/>
mem : 16GB<br/>
os  : ubuntu 18.04<br/>
cuda: 10.2<br/>
cudnn : 7<br/>
python : 3.6.9<br/>
tensorflow-gpu:1.14.0<br/>
numpy : 1.18.1<br/>
opencv-python : 4.1.2.30<br/>
