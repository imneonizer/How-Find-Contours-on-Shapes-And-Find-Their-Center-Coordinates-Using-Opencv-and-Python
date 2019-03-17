import cv2
import numpy as np


img = cv2.imread('shapes.png') #Reading Source Image

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to Graysacale For Efficiency in Computation

contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Finding Contours

print("no of shapes: {0}".format(len(contours))) #Finding No. Of Contours Detected i.e, No. of Shapes

#Drawing Rectangle Contours ========================================
for cnt in contours:
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	img = cv2.drawContours(img, [box], 0, (0,0,255)) #BGR_Color_Sequence
#===================================================================

#finding center of the recognized shapes ===========================
for cnt in contours:
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	center = (cx, cy)
	print("Center coordinate: "+str(center))
#===================================================================

#Drawing polygin Contours ==========================================
	epsilon = 0.01*cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, epsilon, True)
	img = cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
#===================================================================

cv2.imshow('ImageWindow', img) #Showing the Final Image
cv2.waitKey(0)
