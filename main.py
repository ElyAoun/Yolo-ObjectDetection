import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320   # width and height of target
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []


with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # extracts classes from file based on new lines

# print(classNames)  # prints all the classes
# print(len(classNames))  # 80 classes

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # set openCV as our backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, image):
    hT, wT, cT = image.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]  # remove the first 5 elements
            classId = np.argmax(scores)  # index of the max value
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                x, y = int((detection[0]*wT)-w/2), int((detection[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)  # it will tell us which bounding boxes to keep by giving their indices

    for i in indices:
        i = i[0]  # take the first element
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(image, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    _, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)  # convert image to blob
    net.setInput(blob)

    layerNames = net.getLayerNames()  # names of all the layers of the model
    # print(layerNames)

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]  # extract only the output layers

    outputs = net.forward(outputNames)  #  3 output layers
    # print(len(outputs[0].shape))   #  each element is a numpy array
    # print(len(outputs[1].shape))
    # print(len(outputs[2].shape))
    # print(outputs[0][0])  # first element i.e 300x85
    findObjects(outputs, frame)

    cv2.imshow("Live Video", frame)
    cv2.waitKey(1)
