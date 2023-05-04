# import cv2
# import numpy as np


# a = r"yolo\yolov3.cfg"
# b = "yolov3.weights"
# # modelConfiguration = 'cfg/yolov3.cfg'
# # modelWeights = 'yolov3.weights'
# net = cv2.dnn.readNet(a,b)

# classes = []
# with open("classes.txt", "r") as f:
#     classes = f.read().splitlines()

# # cap = cv2.VideoCapture('test1.mp4')

# img = cv2.imread('image.jpg')
# height, width, _ = img.shape

# font = cv2.FONT_HERSHEY_PLAIN
# colors = np.random.uniform(0, 255, size=(100, 3))


# blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
# net.setInput(blob)
# output_layers_names = net.getUnconnectedOutLayersNames()
# layerOutputs = net.forward(output_layers_names)

# boxes = []
# confidences = []
# class_ids = []

# for output in layerOutputs:
#     for detection in output:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.2:
#             center_x = int(detection[0]*width)
#             center_y = int(detection[1]*height)
#             w = int(detection[2]*width)
#             h = int(detection[3]*height)

#             x = int(center_x - w/2)
#             y = int(center_y - h/2)

#             boxes.append([x, y, w, h])
#             confidences.append((float(confidence)))
#             class_ids.append(class_id)

# indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

# if len(indexes)>0:
#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         label = str(classes[class_ids[i]])
#         confidence = str(round(confidences[i],2))
#         color = colors[i]
#         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#         cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

# cv2.imshow('Image', img)
# key = cv2.waitKey(0)


# # cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils

# # Load the image
# image = cv2.imread('test.jpg')

# # Preprocess the image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (416, 416))
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
# image = imagenet_utils.preprocess_input(image)

# # Load the model
# model = load_model('yolov3.h5')

# # Define the class labels
# classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# # Perform object detection
# boxes, scores, labels = model.predict(image)
# for box, score, label in zip(boxes[0], scores[0], labels[0]):
#     if np.any(score > 0.5):
#         xmin = np.float64(box[0] * image.shape[1])
#         ymin = np.float64(box[1] * image.shape[0])
#         xmax = np.float64(box[2] * image.shape[1])
#         ymax = np.float64(box[3] * image.shape[0])
#         cv2.rectangle(image, (float(xmin), float(ymin)), (float(xmax), float(ymax)), (0, 255, 0), 2)
#         cv2.putText(image, classes[label], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

# # Display the image
# cv2.imshow('Object Detection', image)
# cv2.waitKey(0)
