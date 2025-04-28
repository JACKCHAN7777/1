#-----------------------------------------------------------------------#
#   predict.py will provide functions such as single image prediction, camera detection, FPS testing and directory traversal detection.
# Integrate into a py file and modify the pattern by specifying the mode.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the mode of the test:
    #   'predict'           Indicates a single image prediction
    #   'video'             Indicates video detection, you can call the camera or video for detection
    #   'fps'               Indicates the test fps, the image used is street.jpg in img
    #   'dir_predict'       Indicates traversing the folder for detection and saving. By default, iterate through the img folder and save the img_out folder
    #   'heatmap'           Thermogram visualization of the prediction results
    #   'export_onnx'       It means that exporting the model to onnx requires pytorch1.7.1 or above.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                Specifies whether to intercept the target after a single image is predicted
    #   count               Specifies whether to count the target
    #   crop and count are only valid when mode='predict'
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Used to specify the path of the video, when video_path=0 means the detection camera
    #                       If you want to detect video, set such as video_path = "xxx.mp4", which means you can read out the xxx.mp4 file in the root directory.
    #   video_save_path     Indicates the path to save the video. When video_save_path="" means that it is not saved.
    #                       If you want to save the video, set it like video_save_path = "yyy.mp4", which means it is saved as the yyy.mp4 file in the root directory.
    #   video_fps           FPS FOR SAVED VIDEOS
    #
    #   video_path, video_save_path and video_fps are only valid when mode='video'
    # When saving the video, ctrl+c needs to exit or run until the last frame before completing the complete saving step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = "test_video.mp4"
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Used to specify the number of image detection times when measuring fps. Theoretically, the larger the test_interval, the more accurate the fps
    #   fps_image_path      FPS image used to specify test
    #   
    #   test_interval and fps_image_path are only valid when mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Specifies the folder path for the image to be detected
    #   dir_save_path       Specifies the save path for the image that has been detected
    #   
    #   dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   The save path of the heat map is saved under model_data by default
    #   
    #   heatmap_save_path only works in mode='heatmap'
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            Using Simplify onnx
    #   onnx_save_path      Specified the save path of onnx
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        1. If you want to save the detected image, use r_image.save("img.jpg") to save it and modify it directly in predict.py. 
        2. If you want to obtain the coordinates of the prediction box, you can enter the yolo.detect_image function and read the four values of top, left, bottom, and right in the drawing part.
        3. If you want to use the prediction box to intercept the target, you can enter the yolo.detect_image function and use the obtained four values: top, left, bottom, and right in the drawing part.
        The original image is intercepted using matrix.
        4. If you want to write additional words on the prediction graph, such as the number of specific targets detected, you can enter the yolo.detect_image function to judge predicted_class in the drawing part.
        For example, judge if predicted_class == 'car': to determine whether the current target is a car, and then record the number. Use draw.text to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("The camera (video) cannot be read correctly, please note whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # READ A FRAME
            ref, frame = capture.read()
            if not ref:
                break
            # FORMAT CHANGE，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # CONVERT TO IMAGE
            frame = Image.fromarray(np.uint8(frame))
            # PERFORM TESTING
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR satisfies opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
