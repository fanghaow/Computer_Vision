import cv2
import numpy as np
from ellipse import findEllipses

# Read
# filepath = "/home/fanghaow/Computer_Vision/HW2/input/rice.png"
filepath = "/home/fanghaow/Computer_Vision/HW2/input/ellipses.png"
img = cv2.imread(filepath, 0)

# Gaussion Blur
ksize = (5,5)
sigmaX = 2
sigmaY = 2
blur_img = cv2.GaussianBlur(img,ksize,sigmaX,sigmaY) #,cv2.BORDER_DEFAULT)

# Edges extraction
threshold1 = 50
threshold2 = 100
edges = cv2.Canny(blur_img, threshold1, threshold2)

# ellipseMask, contourMask = findEllipses(edges)
ell_img = img.copy()
# ell_img[ellipseMask == 0] = 0
# cont_img = img.copy()
# cont_img[contourMask == 0] = 0

Ellipses = findEllipses(edges)
for ellipse in Ellipses:
    center_coordinates = (int(ellipse[0][0]), int(ellipse[0][1]))
    axesLength = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    angle = ellipse[2]
    startAngle = 0
    endAngle = 360
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 5 px
    thickness = 2
    # Using cv2.ellipse() method
    # Draw a ellipse with red line borders of thickness of 5 px
    ell_img = cv2.ellipse(ell_img, center_coordinates, axesLength,\
            angle, startAngle, endAngle, color, thickness)

'''
# find Contour
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# fitEllipse
X, Y = np.where(edges > 0)
points = np.transpose(np.vstack((X, Y)))
retval = cv2.fitEllipse(points)
center = retval[0]
# axes = 
# ellip_edges = cv2.ellipse(edges, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
'''

# Show
cv2.imshow("Origion", img)
cv2.imshow("Blured", blur_img)
cv2.imshow("Edges", edges)
cv2.imshow("Ellipse", ell_img)
# cv2.imshow("Contour", cont_img)

cv2.waitKey(0)