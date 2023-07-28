import cv2
import numpy as np

#print(cv2.getBuildInformation())

l1 = np.array([[[4,2]],[[2,1]],[[0,1]],[[3,2]]])
print(l1)
print("shape",l1.shape)
l2 = l1.reshape(4,2)
print(l2)
print("shape2",l2.shape)
print("sum",l2.sum(1))
