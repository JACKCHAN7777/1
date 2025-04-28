import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
When training your own object detection model, you must pay attention to the following points:
1. Before training, carefully check whether your format meets the requirements. The library requires that the dataset format is VOC format, and the contents that are prepared include input pictures and labels.
   The input image is a .jpg image, no fixed size is required, and it will be automatically resized before being passed in to training.
   The grayscale image will be automatically converted into RGB image for training without modifying it yourself.
   If the suffix of the input image is not jpg, you need to convert it into jpg in batches before starting training.

   The tag is in .xml format, and there will be target information in the file that needs to be detected, and the tag file corresponds to the input image file.

2. The size of the loss value is used to judge whether it is converged. What is more important is that there is a tendency to converge, that is, the loss of the verification set continues to decline. If the loss of the verification set basically does not change, the model will basically converge.
   The specific size of the loss value is meaningless. The size of the big and small only depends on the calculation method of the loss, and it is not better to be close to 0. If you want to make the loss look good, you can directly divide 10,000 into the corresponding loss function.
   The loss value during training will be saved in the loss_%Y_%m_%d_%H_%M_%S folder in the logs folder
   
3. The trained weight file is saved in the logs folder. Each training generation (Epoch) contains several training steps (Step), and each training step (Step) undergoes gradient descent.
   If you only train a few Steps, you will not save them. You need to clarify the concepts of Epoch and Step.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    WHETHER TO USE CUDA
    #           No GPU can be set to False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    For fixed random seeds
    #           Make each independent training the same result
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use a single machine, multiple cards to run distributedly
    #                   Terminal directives only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify graphics cards under Ubuntu.
    # In Windows system, all graphics cards are called using DP mode by default, and DDP is not supported.
    #   DP mode:
    #       setUp            distributed = False
    #       Enter in the terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode：
    #       setUp            distributed = True
    #       Enter in the terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, DDP mode multi-card is available
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    # Can reduce the video memory by about half, and require pytorch1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    Point to txt under model_data, related to the dataset you trained
    #Be sure to modify classes_path before training to make it correspond to your own data set
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/my_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   The most important part of the model's pre-training weight is the weight part of the backbone feature extraction network, which is used for feature extraction.
    # Pre-training weights must be used for 99% of the cases. If not, the weights of the main part are too random, the feature extraction effect is not obvious, and the results of network training will not be good.
    #
    # If there is an interruption of training operation during training, you can set model_path to the weight file in the logs folder and load the weights that have been trained again.
    # At the same time, modify the parameters of the freezing stage or thawing stage below to ensure the continuity of the model epoch.
    #
    # When model_path = '', the weight of the entire model is not loaded.
    #
    #   Here is the weights of the entire model, so it is loaded in train.py.
    # If you want the model to start training from 0, set model_path = '', and the following Freeze_Train = Fasle, start training from 0, and there is no process of freezing the backbone.
    #
    # Generally speaking, the training effect of the network starting from 0 will be very poor, because the weight is too random and the feature extraction effect is not obvious. Therefore, it is not recommended that everyone start training from 0!
    # There are two solutions to training from 0:
    # 1. Thanks to the powerful data enhancement capability of the Mosaic data enhancement method, when UnFreeze_Epoch is set to be larger (300 and above), batch is larger (16 and above), and data is more (10,000 or above),
    # You can set mosaic=True to start training by randomly initializing the parameters, but the results are still not as good as those in pre-training. (Big datasets like COCO can do this)
    # 2. Understand the imagenet dataset, first train the classification model to obtain the weight of the main part of the network. The main part of the classification model is common to the model, and train based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolov8_s.pth'
    #------------------------------------------------------#
    #   input_shape     The input shape size must be a multiple of 32.
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #------------------------------------------------------#
    #   phi             The version of yolov8 used
    # n: corresponding to yolov8_n
    # s: corresponding to yolov8_s
    # m : corresponding to yolov8_m
    # l: Corresponding to yolov8_l
    # x: corresponding to yolov8_x
    #------------------------------------------------------#
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use the pre-trained weights of the backbone network, the weights of the backbone are used here, so they are loaded during the model construction.
    #                   If model_path is set, the weight of the main trunk does not need to be loaded, and the value of pretrained is meaningless.
    #                   If model_path is not set, pretrained = True, only the backbone is loaded to start training.
    #                   If model_path is not set, pretrained = False, Freeze_Train = Fasle, training starts from 0, and there is no process of freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic              Mosaic data enhancement.
    #   mosaic_prob         How much probability is there for each step to use mosaic data enhancement, default is 50%.
    #
    #   mixup               Whether to use mixup data enhancement, only valid when mosaic=True.
    #                       Only mixup will be performed on images enhanced by mosaic.
    #   mixup_prob          How much probability is there to use mixup data enhancement after mosaic, default is 50%.
    #                       The total mixup probability is mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio   Referring to YoloX, due to the training images generated by Mosaic, it is far from the real distribution of natural images.
    #                       When mosaic=True, this code will enable mosaic within the scope of special_aug_ratio.
    #                       The default is the first 70% of epochs, and 100 generations will be launched for 70 generations.
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing     Label smooth. Generally, below 0.01. Such as 0.01, 0.005.
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two stages, namely the freezing stage and the thawing stage. The freezing phase is set to meet the training needs of students with insufficient machine performance.
    # When the video memory required for freeze training is small and the graphics card is very poor, you can set Freeze_Epoch equal to UnFreeze_Epoch, Freeze_Train = True, and only freeze training is performed at this time.
    #
    # Here are some parameter setting suggestions, and trainers can make flexibly adjustments according to their needs:
    # (I) Start training from the pre-training weights of the entire model:
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       Among them: UnFreeze_Epoch can be adjusted between 100-300.
    #   （二）Start training from 0:
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False
    #       UnFreeze_Epoch Try not to be less than 300 optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freeze phase training parameters
    #     # At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #     # Occupies a small amount of video memory and only fine-tunes the network
    #     # Init_Epoch The current training generation of the model, whose value can be greater than Freeze_Epoch, such as:
    #     #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #     # The freezing phase will be skipped, and the learning rate will be adjusted directly from the 60th generation.
    #     # (Used when resuming a breakpoint)
    #     # Freeze_Epoch The model freezes the Freeze_Epoch of the training
    #     # (Invalid when Freeze_Train=False)
    #     # Freeze_batch_size The model freezes the batch_size of the training
    #     # (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 100 #5
    Freeze_batch_size   = 8
    #------------------------------------------------------------------#
    #   Training parameters in the thawing phase
    #     # At this time, the backbone of the model is not frozen, and the feature extraction network will be changed
    #     # If the video memory is large, all parameters of the network will be changed
    #     # UnFreeze_Epoch The total number of epochs trained by the model
    #     # SGD takes longer to converge, so a larger UnFreeze_Epoch is set
    #     # Adam can use a relatively small UnFreeze_Epoch
    #     # Unfreeze_batch_size batch_size of the model after thawing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300  #15
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether or not to perform a freeze training
    #     # By default, freeze the backbone training first and then unfreeze the training
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model, which defaults to 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  The types of optimizers used can be ADAM and SGD
    #     # When using the Adam optimizer, it is recommended to set Init_lr=1e-3
    #     # When using the SGD optimizer, it is recommended to set Init_lr=1e-2
    #     # The momentum parameter used inside the momentum optimizer
    #     # weight_decay Weight decay to prevent overfitting
    #     # ADAM will cause weight_decay error, and it is recommended to set it to 0 when using ADAM.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   The learning rate reduction methods used are optional step and cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     The number of epochs that save the metric once
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        The folder where the metrics and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether or not to evaluate at the time of training, the evaluation object is the validation set
    #     # After installing the pycocotools library, the evaluation experience is better.
    #     # eval_period represents how many epochs are evaluated at one time, and frequent evaluations are not recommended
    #     # Evaluations take a lot of time, and frequent evaluations can lead to very slow training
    #     # The mAP obtained here will be different from the one obtained by get_map.py for two reasons:
    #     # (1) The mAP obtained here is the mAP of the verification set.
    #     # (2) The evaluation parameters set here are conservative in order to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers     Lets you set whether to use multithreading to read data
    #     # When enabled, it will speed up data reading, but it will take up more memory
    #     # For computers with smaller memory, you can set it to 2 or 0
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path   Training image paths and labels
    #   val_annotation_path     Verify the image path and labels
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    seed_everything(seed)
    #------------------------------------------------------#
    #   Set up the graphics card you are using
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #------------------------------------------------------#
    #   classes and anchor
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #----------------------------------------------------#
    #   Download the pre-training weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)
            
    #------------------------------------------------------#
    #   Create a YOLO model
    #------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load based on the key of the pre-trained weight and the key of the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Shows that there is no match on the key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   Obtain the loss function
    #----------------------#
    yolo_loss = Loss(model)
    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   Torch 1.2 does not support AMP, it is recommended to use Torch 1.7.1 and above to use FP16 correctly
    #     # So torch1.2 here says "could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-card synchronization Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   The total training generation refers to the total number of times all data is traversed
        #         # Total training step size refers to the total number of gradient descent times
        #         # Each training generation contains several training steps, and each training step performs a gradient descent.
        #         # Only the minimum training generation is recommended here, and only the thawing part is considered when calculating
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The dataset is too small to be trained, so enrich the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total amount of training data in this run is %d, the Unfreeze_batch_size is %d, a total of %d epochs are trained, and the total training step size is %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total generation to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   The features of the backbone feature extraction network are universal, and freezing training can speed up the training
    #     # It can also prevent the metric from being destroyed in the early stage of training.
    #     # Init_Epoch is the starting generation
    #     # Freeze_Epoch for the generation of frozen training
    #     # UnFreeze_Epoch total training generation
    #     # If OOM or video memory is insufficient, please reduce the Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze a certain portion of your workout
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If you don't freeze the training, set the batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size and adjust the learning rate adaptively
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Choose an optimizer based on your optimizer_type
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   Get a formula for declining learning rates
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Judge the length of each generation
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, so expand the dataset.")

        if ema:
            ema.updates     = epoch_step * Init_Epoch
        
        #---------------------------------------#
        #   Build a dataset loader.
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   The MAP curve of the EVAL was recorded
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen learning section
            #   then thaw and set the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size and adjust the learning rate adaptively
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get a formula for declining learning rates
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, so expand the dataset.")
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
