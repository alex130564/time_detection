# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:32:21 2023

@author: user
"""

import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

#Step1 檔案讀取
img1 = cv2.imread( "./data/12.jpg", -1 )
#img1 = cv2.imread( "./data/3.png", -1 )
img1 = cv2.resize(img1, (600, 600))
img2 = img1.copy()
img3 = img1.copy()
img4 = img1.copy()

#Step2 圖像預處理
#灰階
gray = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )

#高斯濾波
gaussian_img = cv2.GaussianBlur(gray,(5,5),0)

#中值濾波器
median_img = cv2.medianBlur(gaussian_img,5)

#二進值
ret, bin_img = cv2.threshold(median_img, 90, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
#bin_img = cv2.adaptiveThreshold(gaussian_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                                     cv2.THRESH_BINARY_INV,11,2)

#膨脹
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
dilation = cv2.dilate(bin_img, kernel, iterations = 1)
#cv2.imshow("dilation",dilation)
dilation[dilation==255] = 1

#bin_img[bin_img==255] = 1
#細線化
arms_thin = skeletonize(dilation)
arms_thin = (255*arms_thin).clip(0,255).astype(np.uint8)

#Step3 霍夫直線偵測 + 計算直線長度
linesP = cv2.HoughLinesP(arms_thin , 1, np.pi / 180, 50, None, 60, 80)
lines_length=[]
p1 = []
p2 = []
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(img3, (l[0], l[1]), (l[2], l[3]), (0,255,255), 3, cv2.LINE_AA)
        p1.append(l[0])
        p1.append(l[1])
        p2.append(l[2])
        p2.append(l[3])
        length = int(math.dist(p1,p2))
        lines_length.append(length)
print("linesP=",linesP)
print("lines_length=",lines_length)

#Step4 比較長度，找出時針、分針、秒針
#比較直線長度 
#t = list(np.unit32(length))
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
    minute.append(linesP[max_index[2]][0][i])
    hour.append(linesP[max_index[1]][0][i])

cv2.line(img4, (second[0], second[1]), (second[2], second[3]), (0,255,0), 3, cv2.LINE_AA)
cv2.line(img4, (minute[0], minute[1]), (minute[2], minute[3]), (0,255,255), 3, cv2.LINE_AA)
cv2.line(img4, (hour[0], hour[1]), (hour[2], hour[3]), (0,0,255), 3, cv2.LINE_AA)
print("second(G):",second)
print("minute(Y):",minute)
print("hour(R):",hour)

#Step5 判斷時間
def hour_transfer(a,b,c,d):
    delta_x = c-a
    delta_y = b-d
    print("delta_x=",delta_x)
    print("delta_y=",delta_y)
    #判斷是否垂直
    if delta_x == 0:
        delta_y = -delta_y
    #判斷象限
    if d>=305:
        pi = math.pi
    elif b>=310:
        pi = math.pi
    else:
        pi = 0
    print("pi =",pi)
    #計算角度
    angle = math.degrees(math.atan2(delta_y, delta_x) +pi)
    print("arctan = ",angle)
    #角度轉換成 時鐘的角度
    angle_CCW90 = angle - 90
    if angle_CCW90 < 0:
        angle_CCW90 = angle_CCW90 + 360
    angle_clock = -angle_CCW90 + 360
    #判斷3,6,9,0
    if angle_clock % 90 == 0:
        if delta_y > 0:
            angle_clock = 0
        elif delta_y < 0:
            angle_clock = 180
        elif delta_x > 0:
            angle_clock = 90
        elif delta_x < 0:
            angle_clock =270
    time = math.floor(angle_clock / 30)
    print("time=",time)
    return time

def minute_second_transfer(a,b,c,d):
    delta_x = c-a
    delta_y = b-d
    print("delta_x=",delta_x)
    print("delta_y=",delta_y)
    #判斷是否垂直
    if delta_x == 0:
        delta_y = -delta_y
    #判斷象限
    if d>=305:
        pi = math.pi
    elif b>=310:
        pi = math.pi
    else:
        pi = 0
    print("pi =",pi)
    
    #計算角度
    angle = math.floor(math.degrees(math.atan2(delta_y, delta_x) + pi))
    print("arctan = ",angle)
    #角度轉換成 時鐘的角度
    angle_CCW90 = angle - 90
    if angle_CCW90 < 0:
        angle_CCW90 = angle_CCW90 + 360
    angle_clock = -angle_CCW90 + 360
    #判斷3,6,9,0
    if angle_clock % 90 == 0:
        if delta_y > 0:
            angle_clock = 0
        elif delta_y < 0:
            angle_clock = 180
        elif delta_x > 0:
            angle_clock = 90
        elif delta_x < 0:
            angle_clock =270
    print("angle_clock=",angle_clock)
    time = math.floor(angle_clock / 6)
    print("time=",time)
    return time

time_hour = hour_transfer(hour[0],hour[1],hour[2],hour[3])
time_minute = minute_second_transfer(minute[0],minute[1],minute[2],minute[3])
time_second = minute_second_transfer(second[0],second[1],second[2],second[3])
print("time:%d:%d:%d" %(time_hour,time_minute,time_second))


#cv2.imshow( "Original Image", img1 )
#cv2.imshow( "bin Image", bin_img )
cv2.imshow('arms_thin', arms_thin)
cv2.imshow( "Hough Line DetectionP ", img3 )
cv2.imshow( "minute,hour ", img4 )

cv2.waitKey( 0 )
cv2.destroyAllWindows()

