from pydoc import classname
import cv2
import numpy as np

# openCv DNN
net =cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (320,320), scale = 1/255)

# Load Class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print("Object Lists:")
print(classes)



# initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

def click_button(event, x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print (x,y)
        # polygon = np.array([[(20,20),(220,20),(220,70),(20,70)]]) 

# create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click_button)

while True:
    # get frames***
    ret, frame = cap.read()
    
    # object detection:
    (class_ids,scores,bboxes)=model.detect(frame)
    for class_id, score, bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h) = bbox
        class_name = classes[class_id]
        
        
        cv2.putText(frame, class_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(200,0,50),2)
        
        cv2.rectangle(frame,(x,y), (x+w , y+h), (200,0,50), 3)
        
    # create buttons
    # cv2.rectangle(frame,(20,20),(150,70),(0,0,200),-1)
    polygon = np.array([[(20,20),(220,20),(220,70),(20,70)]]) 
    cv2.fillPoly(frame,polygon,(0,0,200))
    cv2.putText(frame, "Person",(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)
    
        

    cv2.imshow("Frame_Sahil" , frame)

    # cv2.waitKey(0) --> it means that only one frame show ...
    cv2.waitKey(1) #1 means continuous capture frame
    