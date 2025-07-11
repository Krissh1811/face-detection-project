#Importing the necessary packages
import numpy as np
import argparse
import cv2

#Constructing the argument parse and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path ot Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probabilty to filter weak detections")
args = vars(ap.parse_args())

#loading serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#load image and construct input blob for image
#by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

#pass blob through the network and obtain detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

#loop over the detections
for i in range(0, detections.shape[2]):
    #extracting the confidence (probability) associated with prediction
    confidence = detections[0, 0, i, 2]

    #filter out weak detctions by ensuring the `confidence` is greater than minimum confidence
    if confidence > args["confidence"]:
        #compute (x,y) coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #draw the bounding box of face along with the associated probabilty
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 7)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

# Make the window resizable
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# Auto-resize the window to fit within the screen bounds (does NOT stretch the image)
cv2.imshow("Output", image)

# Optional: Resize the window (not the image)
cv2.resizeWindow("Output", 1000, 800)

cv2.waitKey(0)
cv2.destroyAllWindows()


#downloading th eoutput in a separate image
# cv2.imwrite("output.jpg", image)
# print("[INFO] Output image saved as output.jpg")
