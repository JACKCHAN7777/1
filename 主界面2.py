from PyQt5 import QtWidgets, QtGui, QtCore
from ui2 import Ui_MainWindow
from PyQt5.QtWidgets import *
import datetime
import time
from PyQt5.QtGui import QFont
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from waste_category import main as type_inf
from PyQt5 import QtCore
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)


        self.label_type.setText('')
        self.label_pro.setText('')
        self.textEdit.setText('')

        # SET THE FONT SIZE
        font = QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)


        #DATE UPDATE

        today = str(datetime.datetime.now().date())

        self.label_12.setText(today.split('-')[0] + '/' + today.split('-')[1])
        self.label_13.setText(today.split('-')[2])


        self.yolo=YOLO()


        self.timer_camera = QtCore.QTimer()  # Defines a timer that controls the frame rate at which the video is displayed Loop countdown
        self.timer_video = QtCore.QTimer()
        self.cap = cv2.VideoCapture()  # VIDEO STREAMING
        self.CAM_NUM = 0  # A value of 0 indicates that the video stream comes from the laptop's built-in camera

        self.timer_video.timeout.connect(self.display_video_frame)  # IF THE TIMER ENDS THE SHOW_CAMERA IS CALLED
        self.timer_camera.timeout.connect(self.show_camera)  # IF THE TIMER ENDS THE SHOW_CAMERA IS CALLED

        self.pushButton_pic.clicked.connect(self.select_img)  # SELECT IMAGE DETECTION
        self.pushButton_video.clicked.connect(self.select_video)  # SELECT VIDEO DETECTION
        self.pushButton_camera.clicked.connect(self.button_open_camera_clicked)  # SELECT CAMERA DETECTION


        self.name_dict = {
            'Shoes': 'Shoes', 'Egg shells': 'Egg shells','glasses': 'glasses', 'prawns': 'prawns',
            'green vegetables': 'green vegetables', 'napkin': 'napkin','knife': 'knife', 'dolls': 'dolls',
            'watermelon rind': 'watermelon rind', 'power bank': 'power bank', 'plastic bags': 'plastic bags', 'chocolates': 'chocolates',
            'thermometers': 'thermometers', 'milk tea': 'milk tea', 'nappies': 'nappies', 'hot pot base': 'hot pot base',
            'pans': 'pans', 'newspaper': 'newspaper','Egg shellsdddaa': 'Egg shellsdddaa','rice': 'rice',
            'chewing gums': 'chewing gums','cups': 'cups','bulbs': 'bulbs','cigarette': 'cigarette',
            'pill': 'pill','outlet': 'outlet','preservative film': 'preservative film','bags': 'bags','wine bottle': 'wine bottle',
            'meat': 'meat','cans': 'cans','socks': 'socks','lighters': 'lighters','spike': 'spike',
            'banana skin': 'banana skin','breads': 'breads','toothpicks': 'toothpicks','masks': 'masks'
        }


    def select_video(self):
        self.timer_video.stop()  # TURN OFF THE TIMER
        self.timer_camera.stop()  # TURN OFF THE TIMER
        self.cap.release()  # RELEASE THE VIDEO STREAM
        self.label_show.clear()  # CLEAR THE VIDEO DISPLAY AREA
        self.label_show.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                      "border-radius: 20px;")
        self.pushButton_video.setText('Select Video')


        self.file_path, _ = QFileDialog.getOpenFileName(None, 'Select Video file', '', 'Video Files (*.mp4 *.avi)')

        if self.file_path:
            self.pushButton_video.setText('Close/Select Video')
            self.camera = cv2.VideoCapture(self.file_path)
            self.timer_video.start(30)
        else:
            self.label_type.setText('')
            self.label_pro.setText('')
            self.textEdit.setText('')
            self.label_4.setStyleSheet("background-color: rgb(229, 229, 229);")

    def display_video_frame(self):

        flag, self.image = self.camera.read()  # READ FROM THE VIDEO STREAM

        if flag==True:
            show = cv2.resize(self.image, (400, 310))  # RESIZE THE READ FRAME TO 640X480
            cv2.imwrite('./ScreenShot.jpg', show)
            image = Image.open('./ScreenShot.jpg')
            r_image = self.yolo.detect_image(image, crop=False, count=False)
            r_image.save('./res2.png')
            self.label_show.setStyleSheet("image: url(./res2.png)")  # PLACE THE DETECTED IMAGE IN THE DIALOG BOX
            self.show_data()

        else:
            self.timer_video.stop()  # TURN OFF THE TIMER

    def select_img(self):
        self.timer_video.stop()  # TURN OFF THE TIMER
        self.timer_camera.stop()  # TURN OFF THE TIMER
        self.cap.release()  # RELEASE THE VIDEO STREAM
        self.label_show.clear()  # CLEAR THE VIDEO DISPLAY AREA

        self.img_path, _ = QFileDialog.getOpenFileName(None, 'open img', '', "*.png;*.jpg;;All Files(*)")

        if self.img_path:
            print(self.img_path)
            image = Image.open(self.img_path)
            r_image = self.yolo.detect_image(image, crop=False, count=False)
            r_image=r_image.resize((400,310))
            r_image.save('res3.png')
            self.label_show.setStyleSheet("image: url(./res3.png)")  #PLACE THE DETECTED IMAGE IN THE DIALOG BOX

            self.show_data()

    def button_open_camera_clicked(self):
        self.timer_video.stop()  # TURN OFF THE TIMER
        self.label_show.clear()  # CLEAR THE VIDEO DISPLAY AREA
        self.label_show.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                      "border-radius: 20px;")
        if self.timer_camera.isActive() == False:  # IF THE TIMER DOES NOT START

                flag = self.cap.open(self.CAM_NUM)  # If the parameter is 0, the built-in camera of the notebook is turned on, and if the parameter is the video file path, the video is opened
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, 'warning', "Please check whether the camera is connected to the computer correctly",
                                                                    buttons=QtWidgets.QMessageBox.Ok)
                else:
                    self.pushButton_camera.setText("Close camera")
                    self.timer_camera.start(30)  # The timer starts 30ms, and the result is a frame from the camera every 30ms

        else:
            self.timer_camera.stop()  # TURN OFF THE TIMER
            self.cap.release()  # RELEASE THE VIDEO STREAM
            self.label_show.clear()  # CLEAR THE VIDEO DISPLAY AREA
            self.label_show.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                          "border-radius: 20px;")


            self.pushButton_camera.setText('CAMERA DETECTION')
            self.label_type.setText('')
            self.label_pro.setText('')
            self.textEdit.setText('')
            self.label_4.setStyleSheet("background-color: rgb(229, 229, 229);")




    def show_camera(self):

        flag, self.image = self.cap.read()  # READ FROM THE VIDEO STREAM
        show = cv2.resize(self.image, (400, 310))  # RESIZE THE READ FRAME TO 640X480
        cv2.imwrite('./Screenshot.jpg', show)
        image = Image.open('./Screenshot.jpg')
        r_image = self.yolo.detect_image(image, crop=False, count=False)
        r_image.save('./res2.png')
        self.label_show.setStyleSheet("image: url(./res2.png)")  #PLACE THE DETECTED IMAGE IN THE DIALOG BOX
        self.show_data()

    def show_data(self):

        with open('res.txt','r')as fb:
            data=fb.readlines()

        if data:
            for num,i in enumerate(data):
                self.label_type.setText(self.name_dict[i.strip().split('+')[0]])
                self.label_pro.setText(i.strip().split('+')[1]+'%')
                self.textEdit.setText(type_inf(i.strip().split('+')[0])[2])
                if type_inf(i.strip().split('+')[0])[1]=='Other waste':
                    self.label_4.setStyleSheet("image: url(ui/other.jpg)")
                elif type_inf(i.strip().split('+')[0])[1]=='Hazardous waste':
                    self.label_4.setStyleSheet("image: url(ui/Hazardous.jpg)")
                elif type_inf(i.strip().split('+')[0])[1]=='Food waste':
                    self.label_4.setStyleSheet("image: url(ui/food waste.jpg)")
                elif type_inf(i.strip().split('+')[0])[1]=='Recyclables':
                    self.label_4.setStyleSheet("image: url(ui/recycle.jpg)")


        else:
            self.label_type.setText('')
            self.label_pro.setText('')
            self.textEdit.setText('')
            self.label_4.setStyleSheet("background-color: rgb(229, 229, 229);")





if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())