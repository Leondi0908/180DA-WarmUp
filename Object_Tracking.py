# Name: Leondi Soetojo
# ID = 005362251
# Reference:
#


import cv2
import numpy as np

type = ""
while(True):
    type = input("Please enter a type between HSV and RGB to track object:\n")
    if (type == "HSV" or type == "RGB" or type == "hsv" or type == "rgb"):
        break


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while(True):
    ret, img = cap.read()

    if ret == True:        
        
        if (type == "HSV" or type == "hsv"):
            #HSV Value
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Threshold HSV 
            red_lower = np.array([136, 87, 111])
            red_upper = np.array([180, 255, 255])

            # Color Options
            # blue_lower = np.array([99, 115, 150])
            # blue_upper = np.array([110, 255, 255])

            # green_lower = np.array([50, 50, 120])
            # green_upper = np.array([130, 255, 255])
            
            mask = cv2.inRange(hsv, red_lower, red_upper)
            # mask = cv2.inrange(hsv, blue_lower, blue_upper)
            # mask = cv2.inrange(hsv, green_lower, green_upper)
        else:
            #RGB Value
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Threshold HSV 
            red_lower = np.array([128, 0, 0])
            red_upper = np.array([255, 160, 147])

            # Color Options
            # blue_lower = np.array([0, 0, 112])
            # blue_upper = np.array([230, 230, 255])

            # green_lower = np.array([0,100,0])
            # green_upper = np.array([173,255,154])
            
            mask = cv2.inRange(rgb, red_lower, red_upper)
            # mask = cv2.inrange(rgb, blue_lower, blue_upper)
            # mask = cv2.inrange(rgb, green_lower, green_upper)

        res = cv2.bitwise_and(img, img, mask=mask)
        # res = cv2.bitwise_and(img, img, mask= blue)
        # res = cv2.bitwise_and(img, img, mask= green)

        # tracking the Red color
        contours, hierarchy = cv2.findContours(mask, 1, 2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if(area > 1500):
                x, y, w, h = cv2.boundingRect(cnt)
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "RED Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        # (_, contours, hierarchy) = cv2.findcontours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for pic, contour in enumerate (contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x,y,w,h = cv2.boundingRect(contour)
        #         img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        #         cv2.putText(img, "BLUE Color", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

        # (_, contours, hierarchy) = cv2.findcontours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for pic, contour in enumerate (contours):
        #     area = cv2.contourArea(contour)
        #     if(area > 300):
        #         x,y,w,h = cv2.boundingRect(contour)
        #         img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        #         cv2.putText(img, "GREEN Color", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))

        out.write(img)
        cv2.imshow("mask", mask)
        cv2.imshow("image", img)
        cv2.imshow("res", res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if the work is done
cap.release()
out.release()
cv2.destroyAllWindows()
