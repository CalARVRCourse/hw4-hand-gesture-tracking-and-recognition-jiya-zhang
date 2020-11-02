from __future__ import print_function
import cv2
import argparse
import numpy as np
import pyautogui

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Threshold Demo'
isColor = False
pyautogui.FAILSAFE = False

fingerCount = 0
spacePressed = 10

ring_area = -1
pre_ring_area = -1

def extract_hand(frame):
    lower_HSV = np.array([0, 45, 0], dtype = "uint8")  
    upper_HSV = np.array([18, 255, 255], dtype = "uint8")  
      
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
      
      
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")  
          
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
      
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)  
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
      
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask) 
    return skin

def detect_finger_ring(frame):
    global ring_area    
    global pre_ring_area
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  
    
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0  
    #cv2.imshow("labeled img", labeled_img)
    statsSortedByArea = stats[np.argsort(stats[:, 4])]  
    if (ret>2):  
        try:  
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            pre_ring_area = ring_area
            ring_area = w*h
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
            cv2.imshow("ROI "+str(2), subImg)  
            cv2.waitKey(1)  
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
            #print("part2: ", "\n  (x, y): ",(x,y),"\n  (MA, ma): ",(MA,ma),"\n  angle: ",angle)
        except:  
            pre_ring_area = ring_area
            ring_area = -1
            print("No hand found")  
    else:
        pre_ring_area = ring_area
        ring_area = -1
            
def track_fingers(frame):
    global fingerCount
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    ret, thresholdedHandImage = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_OTSU )
    
    _, contours, _ = cv2.findContours(thresholdedHandImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True)     
    thred = cv2.cvtColor(thresholdedHandImage,cv2.COLOR_GRAY2BGR)  
    thredAfterFilter = cv2.cvtColor(thresholdedHandImage,cv2.COLOR_GRAY2BGR)
    if len(contours)>1:  
        largestContour = contours[0]  
        move_with_hand_gesture(largestContour)
        hull = cv2.convexHull(largestContour, returnPoints = False)     
        for cnt in contours[:1]:  
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):  
                fingerCount = 0
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])                   
                    cv2.line(thred,start,end,(0, 255, 0),2) 
                    cv2.circle(thred,far,5,(0, 0, 255),-1)  
                    
                    #partb
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])  
                            
                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
                    
                    if angle <= np.pi / 3:  
                        fingerCount += 1  
                        cv2.circle(thredAfterFilter, far, 4, [0, 0, 255], -1)  
                check_zoom_gestures(defects, cnt)
                cv2.putText(thredAfterFilter, "Finger Counts: " + str(fingerCount+1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("part3 before filtering", thred)  
        cv2.imshow("part3 after filtering", thredAfterFilter)  
    
cX = cY = 0
def move_with_hand_gesture(largestContour):
    global cX
    global cY
    M = cv2.moments(largestContour)  
    cX = 0 + 4  *int(M["m10"] / M["m00"])        
    cY = 0 + 2 *int(M["m01"] / M["m00"])  
    pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)  


def check_space_gestures():
    global fingerCount
    global spacePressed
    if(fingerCount == 4 and spacePressed<=0):       
        pyautogui.press('space')  
        spacePressed = 20      
    else:
        spacePressed -= 1 

escape = False
escape_counter = 0
escape_target_frame = 80

def check_escape_gestures():
    global fingerCount
    global escape
    global escape_counter
    if(fingerCount == 2):       
        escape_counter += 1   
        if(escape_counter >= escape_target_frame):
            escape = True
    else:
        escape_counter = 0

threshold = 100
volume_increase_counter = 0
volume_decrease_counter = 0
def check_volume_up_gestures():
    global ring_area
    global pre_ring_area
    global threshold
    global volume_increase_counter
    global volume_decrease_counter
    if(ring_area > pre_ring_area + threshold):
        #pyautogui.hotkey('fn', 'f5')  
        volume_increase_counter+=1
        volume_decrease_counter = 0
    if(volume_increase_counter > 5):
        pyautogui.press('volumeup')

def check_volume_down_gestures():
    global ring_area
    global pre_ring_area
    global threshold
    global volume_increase_counter
    global volume_decrease_counter
    if(ring_area < pre_ring_area - threshold):
        #pyautogui.hotkey('fn', 'f4')       
        volume_decrease_counter+=1
        volume_increase_counter = 0
    if(volume_decrease_counter > 5):
        pyautogui.press('volumedown') 
click_counter =  0
target_clicks = 50
def check_click_gestures():
    global click_counter
    global target_clicks
    global cX
    global cY
    if (click_counter > target_clicks):
        print("CLICK")
        pyautogui.click(x=cX, y=cY, clicks=1, interval=1, button='left')
        click_counter = 0
    if (fingerCount == 0):
        click_counter += 1

increase_counter = 0
decrease_counter = 0
target_increase_num = 5
target_decrease_num = 5
curr_angle = prev_angle = 0
def check_zoom_gestures(defects, cnt):
    global increase_counter
    global target_increase_num
    global decrease_counter
    global target_decrease_num
    global curr_angle
    global prev_angle
    global fingerCount
    thre = 0.05
    
    # if there is only one detected defect
    # check the angle between the fingers and keep track of it across frames
    if defects is not None and fingerCount == 1:
        s,e,f,d = defects[0,0]  
        start = tuple(cnt[s][0])  
        end = tuple(cnt[e][0])  
        far = tuple(cnt[f][0])  
                
        c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
        a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
        b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
        curr_angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
        #print(decrease_counter, increase_counter, curr_angle)
        if curr_angle < prev_angle - thre:   
            decrease_counter += 1
            increase_counter = max(0, increase_counter-0.5)
            
        elif curr_angle > prev_angle + thre: 
            increase_counter += 1
            decrease_counter = max(0, decrease_counter-0.5)

        prev_angle = curr_angle
        #print(prev_angle)
        if increase_counter > target_increase_num:
            print("ZOOM IN")
            pyautogui.hotkey('ctrl','=')
            increase_counter = 0
        if decrease_counter > target_decrease_num:
            print("ZOOM OUT")
            pyautogui.hotkey('ctrl','-')
            decrease_counter = 0

    
def nothing(x):
    pass
    

cam = cv2.VideoCapture(0)
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_type, window_name , 3, max_type, nothing)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name , 0, max_value, nothing)
# Call the function to initialize
cv2.createTrackbar(trackbar_blur, window_name , 1, 20, nothing)
# create switch for ON/OFF functionality
color_switch = 'Color'
cv2.createTrackbar(color_switch, window_name,0,1,nothing)
cv2.createTrackbar('Contours', window_name,0,1,nothing)


while True:
    ret, frame = cam.read()
    if not ret:
        cv2.destroyAllWindows()
        cam.release()
        break
    

    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
    blur_value = blur_value+ (  blur_value%2==0)
    isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)
    
    frame = extract_hand(frame)

    #convert to grayscale
    if isColor == False:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        if findContours:
            _, contours, hierarchy = cv2.findContours( blur, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
            blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  #add this line
            output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
            print(str(len(contours))+"\n")
        else:
            output = blur
        
        
    else:
        _, dst = cv2.threshold(frame, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        output = blur
    
    detect_finger_ring(frame)
        
    track_fingers(frame)
    check_space_gestures()
    check_escape_gestures()
    check_click_gestures()
    
    check_volume_up_gestures()
    check_volume_down_gestures()
    
    cv2.imshow(window_name, output)

    
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113 or escape:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break
