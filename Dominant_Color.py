# Name: Leondi Soetojo
# ID = 005362251
# Reference:
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
l = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while (True):
    ret, frame = cap.read()

    init_point= (int(l/2), int(h/2))

    x1,y1 = init_point
    # Establish the point for region of interest
    x1 = x1 - (100//2)
    y1 = y1 - (100//2)
    x2 = x1 + 100
    y2 = y1 + 100
    
    # Get the specific frame of region of interest
    img = frame[y1:y2, x1:x2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape the image in the interest frame
    img = img.reshape(img.shape[0] * img.shape[1], 3)

    # Using K-Means Cluster to get the dominant color
    kmeans = KMeans(n_clusters= 3)
    kmeans.fit(img)

    dc ={0:0, 1:0, 2:0}
    # Calculating the dominant color
    for i in kmeans.labels_:
        dc[i] = dc[i] + 1

    dominant_val = 0
    dom = int
    for key, value in dc.items():
        if(value > dominant_val):
            dominant_val = value
            dom = key

    # Getting the dominant color and turn into integer
    rect_color = kmeans.cluster_centers_[dom]
    rect_color = [int (i) for i in rect_color]
    rect_color.reverse()
    
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2, 8, 0)
    cv2.putText(frame, "Dominant Color: {}". format(rect_color), (x1, y1), 0, 0.7, rect_color)
    
    out.write(frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()




