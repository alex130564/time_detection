# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:06:09 2023

@author: user
參考：https://stackoverflow.com/questions/67537837/how-to-detect-clock-hands-with-hough-lines-detection
"""

import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

#Step1 檔案讀取
img1 = cv2.imread( "./data/2.jpg", -1 )
img1 = cv2.resize(img1, (600, 600))
img2 = img1.copy()
img3 = img1.copy()
img4 = img1.copy()
#Step2 圖像預處理
#灰階
gray = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )

#質方圖等化
clahe = cv2.createCLAHE(clipLimit=2)
clahe_img = clahe.apply(gray)

#高斯濾波
gaussian_img = cv2.GaussianBlur(clahe_img,(5,5),0)

#中值濾波器
median_img = cv2.medianBlur(gaussian_img,3)

#二進值
ret, bin_img = cv2.threshold(median_img, 90, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)

#Step3 遮罩，提取指針
#尋找輪廓
#canny_img = cv2.Canny(bin_img, 20, 160)
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#計算面積，界在自訂的值內(2000<area<7000)才畫出輪廓
mask_img = np.zeros_like(bin_img)
for i, j in enumerate(contours):
    area = int(cv2.contourArea(contours[i]))
    #print("area=",area)
    if area>2200 and area<7000:
        cv2.drawContours(mask_img, contours, i, (255,255,255), -1)

#骨架提取
mask_img[mask_img==255] = 1
arms_thin = skeletonize(mask_img)
arms_thin = (255*arms_thin).clip(0,255).astype(np.uint8)

#Step4 直線偵測
linesP = cv2.HoughLinesP(arms_thin , 1, np.pi / 180, 40, None, 10, 25) #40 10 20' 40 10 25
print(linesP)
lines_length=[]
p1 = []
p2 = []
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(img3, (l[0], l[1]), (l[2], l[3]), (0,255,255), 2, cv2.LINE_AA)
        p1.append(l[0])
        p1.append(l[1])
        p2.append(l[2])
        p2.append(l[3])
        length = int(math.sqrt((l[0]-l[2])**2+(l[1]-l[3])**2))
        #length = int(math.dist(p1,p2))
        lines_length.append(length)


#Step5 區分指針
#比較直線長度 
t = lines_length
max_number = []
max_index = []
for _ in range(3):
    number = max(t)
    index = t.index(number)
    t[index] = 0
    max_number.append(number)
    max_index.append(index)
print(max_number)
print(max_index)

#劃出分針 時針
minute = []
hour = []
second = []
for i in range(4):
    second.append(linesP[max_index[0]][0][i])
    minute.append(linesP[max_index[1]][0][i])
    hour.append(linesP[max_index[2]][0][i])

cv2.line(img4, (second[0], second[1]), (second[2], second[3]), (0,255,0), 3, cv2.LINE_AA)
cv2.line(img4, (minute[0], minute[1]), (minute[2], minute[3]), (0,255,255), 3, cv2.LINE_AA)
cv2.line(img4, (hour[0], hour[1]), (hour[2], hour[3]), (0,0,255), 3, cv2.LINE_AA)
print("second(G):",second)
print("minute(Y):",minute)
print("hour(R):",hour)

#Step6 判斷時間
def claculate_time(a,mode):
    print(a)
    a[0] =  a[0]-300
    a[1] = -a[1]+300
    a[2] =  a[2]-300
    a[3] = -a[3]+300
    print("校正:",a)
    delta_x = a[2]-a[0]
    delta_y = a[3]-a[1]
    print("delta_x=",delta_x)
    print("delta_y=",delta_y)
    #判斷象限
    if delta_y/delta_x > 0 and a[0]<-40:
        pi=math.pi
    elif delta_y/delta_x < 0 and a[0]<-50 and a[2]<35:
        pi=math.pi
    else:
        pi=0
    print("pi=",pi)
    #計算角度
    angle = math.degrees(math.atan2(delta_y,delta_x) + pi) 
    if angle <0 :
        angle=angle+360
    print("arctan=",angle)
    #轉換角度座標
    angle_CCW90 = angle - 90
    if angle_CCW90 < 0:
        angle_CCW90 = angle_CCW90 + 360
    angle_clock = -angle_CCW90 + 360
    #如果最後角度>360，扣一圈(360
    if angle_clock>360:
        angle_clock=angle_clock-360
    print("angle_clock = ",angle_clock)

    #判斷3,6,9,0
    if angle_clock % 90 == 0:
        if delta_y > 0:
            angle_clock = 0
        elif delta_y < 0:
            angle_clock = 180
        elif delta_x < 0:
            angle_clock = 90
        elif delta_x > 0:
            angle_clock =270
    #計算時間
    if mode==1:
        time = math.floor(angle_clock / 6)
    elif mode ==0:
        time = math.floor(angle_clock / 30)
    print("time=",time)
    return time


time_hour = claculate_time(hour,0)
time_minute = claculate_time(minute,1)
time_second = claculate_time(second,1)
print("time:%d:%d:%d" %(time_hour,time_minute,time_second))
cv2.putText(img4, "time:%d:%d:%d" %(time_hour,time_minute,time_second), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (180, 0, 0), 1, cv2.LINE_AA)
#plt.hist(clahe_img.ravel(), 256, [0, 256])
#plt.show()
cv2.imshow("clahe_img",clahe_img)
cv2.imshow("bin_img",bin_img)
#cv2.imshow("img_contours",mask_img)
cv2.imshow('arms_thin', arms_thin)
cv2.imshow( "Hough Line DetectionP ", img3 )
cv2.imshow( "minute,hour ", img4 )
cv2.imwrite("minute,hour.jpg ", img4)
cv2.waitKey( 0 )
cv2.destroyAllWindows()