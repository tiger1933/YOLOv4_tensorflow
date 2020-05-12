# YOLOv4_tensorflow
yolov4的tensorflow实现. <br/>
Implement yolov4 with tensorflow<br/>
持续更新<br/>
continuous update the code</br>
</br>

## 以前代码的 mish 激活函数实现错误，请一定要用最新版本的代码。</br>
## The previous work about MISH activation function was wrong, please use the latest code.</br>
</br>

## 二战中科院计算所失败，求老师调剂收留
## The second time I failed in the postgraduate entrance examination, I eager the teacher would accept me.
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
if have no error, it's ok</br>
</br>

### 转换 yolov4.weights
### convert yolov4.weights to fit our code
参考[这个权重转换文件](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), 我将 yolov4.weights 转换到了自己的代码中,执行命令<br/>
refer to [this weights convert file](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/convert_weight.py), i converted yolov4.weights to this project.</br>
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
你将会看到这样的画面,完美</br>
you'll see ..., perfect</br>
![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/coco_save/dog.jpg)
</br>
weights_name.txt 文件中存放的是图模型的卷积层和bn的名字<br/>
the weights_name.txt contains all model layer's name of the network <br/>
<br/>
数据增强策略没有实现，以后会慢慢更新</br>
data enhancement strategies have not been implemented, and will be updated slowly in the future.</br>
<br/>

### 在 VOC 数据集上训练
### train on VOC
打开 config.py ,将 voc_root_dir 修改为自己VOC数据集存放的根目录, voc_dir_ls 修改为自己想要训练的VOC数据集名</br>
open config.py and modify the voc_root_dir to the root of your VOC dataset, modiry the voc_dir_ls to the name of VOC dataset witch  you want to train </br>
</br>
执行命令</br>
run the command</br>
```
python train_voc.py
```
训练完成后,将测试图片放到 voc_test_pic 文件夹下,执行命令</br>
put test images into voc_test_pic folder, and run the following command after the model training.</br>
```
python val_voc.py
```
</br>
训练一晚上的结果，也还可以</br>
it's the result of our code training for a night, not bad</br>
![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/voc_save/000302.jpg)
</br>
此外，在VOC上训练时，我发现损失有时候会是Nan, 我正在尝试解决这个问题.</br>
in addition, i found that the loss sometimes become Nan  when i training on VOC, i'm trying to repair this bug.</br>

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
open config.py, write the anchor box to line 6, just like this<br/>
anchors = 12,19, 19,27, 18,37, 21,38, 23,38, 26,39, 31,38, 39,44, 67,96<br/>
</br>
接下来，修改 data/train.names 中的内容为你需要训练的分类名字(不要用中文),并且将 config.py 中的分类数改为自己的分类数</br>
and, now, modify the content in data/train.names to the category name that you need to train, and change the class_num in config.py to your own category number.</br>
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
这是我用123张图片训练了 5000 步(25分钟)的结果，效果还不错<br/>
this image is the result of training 5000 steps (25 minutes) with 123 pictures, it looks not bad. <br/>
![image](https://github.com/rrddcc/YOLOv4_tensorflow/blob/master/save/62.jpg)

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
