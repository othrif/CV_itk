import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('.\CornerDots.tif')
imgMatch = cv2.imread('.\R05.jpg')

# Initiate STAR detector
orb = cv2.ORB_create(nfeatures=1000, patchSize=7, scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints with ORB
kp = orb.detect(img,None)
kpMatch = orb.detect(imgMatch,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
kpMatch, desMatch = orb.compute(imgMatch, kpMatch)

img2=img
imgMatch2=imgMatch

# draw only keypoints location,not size and orientation
cv2.drawKeypoints(img,kp,img2,color=(0,255,0), flags=0)
cv2.drawKeypoints(imgMatch,kpMatch,imgMatch2,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()

plt.imshow(imgMatch2),plt.show()


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des,desMatch)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img,kp,imgMatch,kpMatch,matches,None, flags=2)
plt.imshow(img3)
plt.show()
