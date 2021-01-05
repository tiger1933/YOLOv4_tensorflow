# coding:utf-8
# configuration file

# ############# Basic configuration. #############
size = width = height = 416                     # image size
batch_size = 2
total_epoch = 45       # total epoch
save_per_epoch = 5        # per save_step save one model
data_debug = False       # load data in debug model (show pictures when loading images)
cls_normalizer = 1.0    # Loss coefficient of confidence
iou_normalizer = 0.07   # loss coefficient of ciou
iou_thresh = 0.5     # 
prob_thresh = 0.25      # 
score_thresh = 0.25     # 
val_score_thresh = 0.5      # 
val_iou_thresh = 0.213            # 
max_box = 50                # 
save_img = False             # save the result image when test the net

# ############# log #############
log_dir = './log'
log_name = 'log.txt'
loss_name = 'loss.txt'

# configure the leanring rate
lr_init = 2e-4                      # initial learning rate	# 0.00261
lr_lower =1e-6                  # minimum learning rate    
lr_type = 'piecewise'   # type of learning rate( 'exponential', 'piecewise', 'constant')
piecewise_boundaries = [3, 90, 100]   #  for piecewise
piecewise_values = [2e-4, 0.0032, 2e-4, 1e-4]   # piecewise learning rate

# configure the optimizer
optimizer_type = 'momentum' # type of optimizer
momentum = 0.949          # 
weight_decay = 0.0005

# ############## train on VOC ##############
voc_root_dir = ["/media/random/数据集/数据集/VOC/VOC2007",
                                    "/media/random/数据集/数据集/VOC/VOC2012"]  # root directory of voc dataset
voc_test_dir = "./voc_test_pic"                                                 # test pictures directory for VOC dataset
voc_save_dir = "./voc_save"                                                     # the folder to save result image for VOC dataset
voc_model_path = "./VOC"                                                        # the folder to save model for VOC dataset
voc_model_name = "voc"                                          # the model name for VOC dataset
voc_names = "./data/voc.names"                             # the names of voc dataset
voc_class_num = 20
voc_anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
