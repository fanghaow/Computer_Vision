import cv2
import numpy as np

def findEllipses(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ellipseMask = np.zeros(edges.shape, dtype=np.uint8)
    contourMask = np.zeros(edges.shape, dtype=np.uint8)

    pi_4 = np.pi * 4
    Ellipses = []

    for i, contour in enumerate(contours):
        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)
        if area <= 100:  # skip ellipses smaller then 10x10
            continue

        arclen = cv2.arcLength(contour, True)
        circularity = (pi_4 * area) / (arclen * arclen)
        ellipse = cv2.fitEllipse(contour)
        Ellipses.append(ellipse)
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)

        # if contour is circular enough
        circular_th = 0.1
        if circularity > circular_th:
            cv2.fillPoly(ellipseMask, [poly], 255)
            continue

        # if contour has enough similarity to an ellipse
        # similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, cv2.cv2.CV_CONTOURS_MATCH_I2, 0)
        # if similarity <= 0.2:
        #     cv2.fillPoly(contourMask, [poly], 255)

    # return ellipseMask, contourMask 
    return Ellipses