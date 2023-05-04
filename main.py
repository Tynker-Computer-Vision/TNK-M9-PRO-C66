import numpy as np
import cv2

# Set confidence and NMS thresholds
confidenceThreshold = 0.5
NMSThreshold = 0.3

# Path to model configuration and weights files
modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

# Path to labels file
labelsPath = 'coco.names'

# Load labels from file
labels = open(labelsPath).read().strip().split('\n')

# Set random seed and generate random colors for each class
np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Load YOLO object detection network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Load image
image = cv2.imread('static/img3.jpg')

# Get image dimensions
(H, W) = image.shape[:2]

# Create blob from image and set input for YOLO network
blob = cv2.dnn.blobFromImage(
    image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Get names of unconnected output layers
layerName = net.getUnconnectedOutLayersNames()

# Forward pass through network
layerOutputs = net.forward(layerName)

# Initialize lists to store bounding boxes, confidences, and class IDs
boxes = []
confidences = []
classIDs = []

# Process each output from YOLO network
for output in layerOutputs:
    for detection in output:
        # Get class scores and ID of class with highest score
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # If confidence threshold is met, save bounding box coordinates and class ID
        if confidence > confidenceThreshold:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY,  width, height) = box.astype('int')
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply Non Maxima Suppression to remove overlapping bounding boxes
detectionNMS = cv2.dnn.NMSBoxes(
    boxes, confidences, confidenceThreshold, NMSThreshold)

# If at least one detection remains after NMS
if (len(detectionNMS) > 0):
    # Process each remaining detection
    for i in detectionNMS.flatten():
        # Get bounding box coordinates and class color
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]

        # Draw bounding box and label on image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        text = '{}: {:.2f}'.format(labels[classIDs[i]], confidences[i]*100)
        cv2.putText(image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Display image with bounding boxes and labels
cv2.imshow('Image', image)
cv2.waitKey(0)
