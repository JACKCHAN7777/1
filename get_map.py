import glob
import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map, voc_ap, file_lines_to_list
from confusion_matrix import run_image_level_confusion_matrix
from yolo import YOLO



if __name__ == "__main__":
    '''
   Recall and Precision are not the same as AP as the concept of area, so the Recall and Precision values of the network are different at the same time as the threshold value (Confidence).
    By default, the Recall and Precision calculated by this code represent the corresponding Recall and Precision values when the threshold value (Confidence) is 0.5.

    Due to the limitations of mAP calculation principle, the network needs to obtain nearly all prediction boxes when calculating mAP, so that the Recall and Precision values under different threshold conditions can be calculated.
    Therefore, the number of txt boxes in map_out/detection-results/ obtained in this code will generally be more than that in direct prediction, and the purpose is to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    # map_mode is used to specify the content calculated when the file is running
    # map_mode is 0 to represent the entire map calculation process, including obtaining prediction results, obtaining the real box, and calculating VOC_map.
    # map_mode is 1 to only obtain the prediction result.
    # map_mode is 2 means that only the real box is obtained.
    # map_mode is 3 means that only VOC_map is calculated.
    # map_mode is 4. The COCO toolbox is used to calculate the current dataset's 0.50:0.95map. You need to obtain the prediction results, obtain the real box and install pycocotools
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #  The classes_path here is used to specify the categories that need to measure VOC_map
    # Generally, it can be consistent with the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/my_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP is used to specify the mAP0.x you want to obtain. What is the meaning of mAP0.x, please Baidu.
    # For example, to calculate mAP 0.75, you can set MINOVERLAP = 0.75.
    #
    # When a prediction box coincides with the real box greater than MINOVERLAP, the prediction box is considered a positive sample, otherwise it is a negative sample.
    # Therefore, the larger the value of MINOVERLAP, the more accurate the prediction box must predict to be considered a positive sample. The lower the calculated mAP value at this time.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   Due to the limitations of mAP calculation principle, the network needs to obtain nearly all prediction boxes when calculating mAP, so that mAP can be calculated.
    # Therefore, the value of confidence should be set as small as possible to obtain all possible prediction boxes.
    #
    # This value is not generally adjusted. Because calculating mAP requires obtaining nearly all prediction boxes, the confidence here cannot be changed casually.
    # To obtain Recall and Precision values under different threshold values, please modify score_threhold below.
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   The size of the non-maximum suppression value used during prediction is that the larger the non-maximum suppression, the less stringent it is.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   REcall and Precision are not the same as AP as the concept of area, so when the threshold values are different, the Recall and Precision values of the network are different.
    #
    # By default, the Recall and Precision calculated by this code represent the Recall and Precision values corresponding to the threshold value is 0.5 (defined here as score_threhold).
    # Because calculating mAP requires obtaining nearly all prediction boxes, the confidence defined above cannot be changed at will.
    # Here we specifically define a score_threhold to represent the threshold value, and then find the Recall and Precision values corresponding to the threshold value when calculating mAP.
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis is used to specify whether to enable VOC_map calculations to be enabled
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   Point to the folder where the VOC dataset is located
    # By default, point to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   The folder output of the result is map_out by default
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")

    run_image_level_confusion_matrix(
        gt_path=os.path.join(map_out_path,'ground-truth'),
        pred_path=os.path.join(map_out_path,'detection-results'),
        class_names=class_names,
        map_out_path=map_out_path
    )
