# coding:utf-8
# configuration file

# ############# Basic configuration. #############
class_num = 25
anchors = 12,19, 14,36, 20,26, 20,37, 23,38, 27,39, 32,35, 39,44, 67,96
model_path = "./checkpoint/"
model_name = "model"
name_file = './data/train.names'                # dataset's classfic names

# ############# log #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# ############## train ##############
train_file = './data/train.txt'
batch_size = 2
multi_scale_img = False     # mutiscale zoom image to training
keep_img_shape = False              # keep image shape when we resize the image
flip_img = True                # flip the image vertically
gray_img = True             # data augment of make image to gray
label_smooth = False     #do not set it as true. label smooth . maybe it have some trouble in training in our code
erase_img = True            # random erase image  
invert_img = True                       # reverse the image pixels (1.0 - pixels.value)          
rotate_img = False               # rotate image and do not set it as True
data_augment = [False, False, flip_img, gray_img, False, erase_img, False, False] # data augment
total_epoch = 300       # total epoch
save_step = 30000        # per save_step save one model
data_debug = False       # load data in debug model

cls_normalizer = 1.0    # Loss coefficient of confidence
ignore_thresh = 0.7     # 
prob_thresh = 0.25      # 
score_thresh = 0.25     # 

# configure the leanring rate
lr_init = 2e-4                      # initial learning rate	# 0.00261
lr_lower =1e-6                  # minimum learning rate    
lr_type = 'constant'   # type of learning rate( 'exponential', 'piecewise', 'constant')
piecewise_boundaries = [1, 2]   #  for piecewise
piecewise_values = [2e-4, 1e-4, 1e-4]   # piecewise learning rate

# configure the optimizer
optimizer_type = 'momentum' # type of optimizer
momentum = 0.949          # 
weight_decay = 0.0005


# ############## test ##############
val_score_thresh = 0.5      # 
iou_thresh = 0.213            # 
max_box = 50                # 
val_dir = "./test_pic"  # Test folder directory, which stores test pictures
save_img = False             # save the result image when test the net
save_dir = "./save"         # the folder to save result image
width = 416                     # image width in net
height = 416                    # image height in net


# ############## train on VOC ##############
voc_root_dir = "/home/random/下载/VOC_dataset"  # root directory of voc dataset
voc_dir_ls = ['2007_trainval', '2012_trainval']                # the version of voc dataset
voc_test_dir = "./voc_test_pic"                                                 # test pictures directory for VOC dataset
voc_save_dir = "./voc_save"                                                     # the folder to save result image for VOC dataset
voc_model_path = "./VOC"                                                        # the folder to save model for VOC dataset
voc_model_name = "voc"                                          # the model name for VOC dataset
voc_names = "./data/voc.names"                             # the names of voc dataset
voc_class_num = 20
voc_anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
