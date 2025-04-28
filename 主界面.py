from PyQt5 import QtWidgets, QtGui, QtCore
from tensorboard.summary.v1 import image

from ui import Ui_MainWindow
from PyQt5.QtWidgets import *
import datetime
import time

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
        self.detect_interval = 3


        self.label_type.setText('')
        self.label_pro.setText('')
        self.label_time.setText('')
        self.label_num.setText('')

        #Date Update

        today = str(datetime.datetime.now().date())

        self.label_12.setText(today.split('-')[0] + '/' + today.split('-')[1])
        self.label_13.setText(today.split('-')[2])


        # Setting the table adaptive size
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        #Setting the number of rows and columns in a table
        self.tableWidget.setRowCount(7)
        self.tableWidget.setColumnCount(6)

        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        #Columns adaptive, (horizontally occupying the windows)
        self.tableWidget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        #Row Adaptive, (vertical orientation fills the windows)


        #Column headings are hidden
        self.tableWidget.verticalHeader().hide()  #列

        #Setting the table header
        self.tableWidget.setHorizontalHeaderLabels(['No.', 'File Paths', 'Result','Confidence','Classification','Location'])

        #Show gridlines
        self.tableWidget.setShowGrid(True)





        # Click on the event to get the selected content, rows and columns
        self.tableWidget.cellPressed.connect(self.getPosContent)


        self.yolo=YOLO()





        self.timer_camera = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0

        self.timer_video.timeout.connect(self.display_video_frame)
        self.timer_camera.timeout.connect(self.show_camera)

        self.pushButton_pic.clicked.connect(self.select_img)  # Select Image Detection
        self.pushButton_video.clicked.connect(self.select_video)  # Select Video Detection
        self.pushButton_camera.clicked.connect(self.button_open_camera_clicked)  # Select Camera Detection

        self.detect_flag=0
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
        self.timer_video.stop()
        self.timer_camera.stop()
        self.cap.release()
        self.label_show.clear()
        self.label_show.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                      "border-radius: 20px;")
        self.tableWidget.clearContents()
        self.pushButton_video.setText('Select Video')


        self.file_path, _ = QFileDialog.getOpenFileName(None, 'Select Video file', '', 'Video Files (*.mp4 *.avi)')

        if self.file_path:
            self.pushButton_video.setText('Close/Select Video')
            self.detect_flag=1
            self.camera = cv2.VideoCapture(self.file_path)
            self.timer_video.start(33)
        else:
            self.label_type.setText('')
            self.label_pro.setText('')
            self.label_time.setText('')
            self.label_num.setText('')

    def display_video_frame(self):


        flag, self.image = self.camera.read()  # Read from video stream

        if flag==True:
            show = cv2.resize(self.image, (640, 480))# RESET THE SIZE OF THE READ FRAME TO 640X480
            cv2.imwrite('./Screenshot.jpg', show)
            image = Image.open('./Screenshot.jpg')
            r_image = self.yolo.detect_image(image, crop=False, count=False)
            r_image.save('./res2.png')
            self.label_show.setStyleSheet("image: url(./res2.png)")  # Put the detected image into the interface box
            self.show_data()

        else:
            self.timer_video.stop()  # TURN OFF THE TIMER


    def select_img(self):
        self.detect_flag=0
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
            self.label_show.setStyleSheet("image: url(./res3.png)")  #PUT THE DETECTED IMAGE INTO THE INTERFACE BOX

            self.show_data()

    def button_open_camera_clicked(self):
        self.timer_video.stop()  # TURN OFF THE TIMER
        self.label_show.clear()  # CLEAR THE VIDEO DISPLAY AREA
        self.label_show.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                      "border-radius: 20px;")
        if self.timer_camera.isActive() == False:  # IF THE TIMER IS NOT STARTED
                self.detect_flag=2
                flag = self.cap.open(self.CAM_NUM)  # The parameter is 0, which means that the built-in camera of the notebook is turned on. If the parameter is the video file path, the video will be turned on.
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, 'warning', "Please check whether the camera is connected correctly to the computer",
                                                                    buttons=QtWidgets.QMessageBox.Ok)
                else:
                    self.pushButton_camera.setText("TURN OFF THE CAMERA")
                    self.timer_camera.start(30)  # The timer starts to time for 30ms, and the result is that every 30ms you have to take a frame from the camera to display

        else:
            self.timer_camera.stop()  # TURN OFF THE TIMER
            self.cap.release()  # RELEASE THE VIDEO STREAM
            self.label_show.clear()  # CLEAR THE VIDEO DISPLAY AREA
            self.label_show.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                          "border-radius: 20px;")
            self.tableWidget.clearContents()

            self.pushButton_camera.setText('CAMERA DETECTION')
            self.label_type.setText('')
            self.label_pro.setText('')
            self.label_time.setText('')
            self.label_num.setText('')




    def show_camera(self):

        flag, self.image = self.cap.read()  # READ FROM VIDEO STREAM
        show = cv2.resize(self.image, (400, 310))  # RESET THE SIZE OF THE READ FRAME TO 640X480
        cv2.imwrite('./Screenshot.jpg', show)
        image = Image.open('./Screenshot.jpg')
        r_image = self.yolo.detect_image(image, crop=False, count=False)
        r_image.save('./res2.png')
        self.label_show.setStyleSheet("image: url(./res2.png)")  #PUT THE DETECTED IMAGE INTO THE INTERFACE BOX
        self.show_data()

    def show_data(self):
        self.tableWidget.clearContents()
        self.excel_data=[]
        with open('time.txt','r')as fb:
            time_data=fb.read()
        self.label_time.setText(time_data)
        with open('res.txt','r')as fb:
            data=fb.readlines()
        self.label_num.setText(str(len(data)))

        self.tableWidget.setRowCount(len(data))

        if data:
            for num,i in enumerate(data):
                self.label_type.setText(self.name_dict[i.strip().split('+')[0]])
                self.label_pro.setText(i.strip().split('+')[1]+'%')
                if self.detect_flag==0:
                    path=self.img_path
                elif self.detect_flag==1:
                    path=self.file_path
                elif self.detect_flag==2:
                    path='camera'
                self.excel_data.append(
                    [str(num+1),path,self.name_dict[i.strip().split('+')[0]],
                     i.strip().split('+')[1]+'%',type_inf(i.strip().split('+')[0])[1],
                     [i.strip().split('+')[2],i.strip().split('+')[3],i.strip().split('+')[4],i.strip().split('+')[5]]
                     ]
                )

            for row, item in enumerate(self.excel_data):
                for column, data in enumerate(item):
                    self.tableWidget.setItem(row, column, QtWidgets.QTableWidgetItem(str(data)))
        else:
            self.label_type.setText('')
            self.label_pro.setText('')



    # GET SELECTED RANKS COLUMNS CONTENTS
    def getPosContent(self, row, col):
        try:
            content = self.tableWidget.item(row, col).text()
            print("SELECT THE ROW：" + str(row))
            print('SELECT CONTENT:' + content)
            print(self.excel_data[row][3])

            self.label_type.setText(self.excel_data[row][2])
            self.label_pro.setText(self.excel_data[row][3])


        except:
            print('Selected content is empty')



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())