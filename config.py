# coding:utf-8
# 配置文件

# ############# 基本配置 #############
class_num = 25
anchors = 12,19, 14,36, 20,26, 20,37, 23,38, 27,39, 32,35, 39,44, 67,96
model_path = "./checkpoint/"
model_name = "model"
name_file = './data/train.names'                # 自己的数据集的名字

# ############# 日志 #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# ############## 训练 ##############
train_file = './data/train.txt'
batch_size = 2
multi_scale_img = False     # 多尺度缩放图片训练
keep_img_shape = True              # resize时保持图片形状
flip_img = True                # 翻转图片
gray_img = True             # 灰度化图片
label_smooth = True     # 标签平滑  
erase_img = True            # 随机擦除  
invert = True                       # 图片像素取反           
data_augment = [multi_scale_img, keep_img_shape, flip_img, gray_img, label_smooth, erase_img, invert] # 数据增强策略
total_epoch = 300       # 一共训练多少 epoch
save_step = 5000        # 多少步保存一次

cls_normalizer = 1.0    # 置信度损失系数
ignore_thresh = 0.7     # 与真值 iou / giou 小于这个阈值就认为没有预测物体
prob_thresh = 0.25      # 分类概率的阈值
score_thresh = 0.25     # 分类得分阈值

# 学习率配置
lr_init = 2e-4                      # 初始学习率	# 0.00261
lr_lower =1e-6                  # 最低学习率    
lr_type = 'constant'   # 学习率类型 'exponential', 'piecewise', 'constant'
piecewise_boundaries = [1, 2]   # 单位:epoch, for piecewise
piecewise_values = [2e-4, 1e-4, 1e-4]

# 优化器配置
optimizer_type = 'momentum' # 优化器类型
momentum = 0.949          # 动量
weight_decay = 0.0005


# ############## 测试 ##############
val_score_thresh = 0.5      # 少于这个分数就忽略
iou_thresh = 0.213            # iou 大于这个值就认为是同一个物体
max_box = 50                # 物体最多个数
val_dir = "./test_pic"  # 测试文件夹, 里面存放测试图片
save_img = True             # 是否保存测试图片
save_dir = "./save"         # 图片保存路径
width = 416                     # 图片宽, 6G显存跑不起来 608 的, 有更好的显卡可以跑608
height = 416                    # 图片高


# ############## VOC训练 ##############
voc_root_dir = "/home/random/下载/VOC_dataset"  # voc 数据集存放的根目录
voc_dir_ls = ['2007_trainval', '2012_trainval']                # 使用的voc数据集名字
voc_test_dir = "./voc_test_pic"                                                 # voc 数据集的测试图片
voc_save_dir = "./voc_save"                                                     # voc 数据集保存的图片
voc_model_path = "./VOC"                                                        # voc 模型保存路径
voc_model_name = "voc"                                          # voc 训练保存的模型名字
voc_names = "./data/voc.names"                             # voc 物体名
voc_class_num = 20
voc_anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
