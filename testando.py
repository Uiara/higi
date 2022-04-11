import cv2 
import os

def opencv_dnn_face_location(filename, path_dnn_face_detector, miniSize):
    # load image from file
    image = cv2.imread(filename)
    #face detection confidence. All face detections under this value are discarded
    confidence_ = 0.65
    
    #list to (x, y)-coordinates of the bounding box for the face
    boxes = []
    face_locations = []
    #path to files needed to face detector
    protoPath = os.path.sep.join([path_dnn_face_detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([path_dnn_face_detector, "res10_300x300_ssd_iter_140000.caffemodel"])

    #load files to face detector
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    #grab the image dimensions
    (h, w) = image.shape[:2]

    #construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0))

    #apply OpenCV's deep learning-based face detector to localize
    #faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    print(detections.shape)
    #loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the confidence (i.e., probability) associated with the
        #prediction
        confidence = detections[0, 0, i, 2]
        #print(confidence, "confidence")
        #filter out weak detections
        if confidence > confidence_:
            #compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if (startY <= 1.1*h and endY <= 1.1*h and endX <= 1.1*w and startX <= 1.1*w):
                #reordering coordenates to use in face recognition
                boxes.append((startY, endX, endY, startX))
                for (startY, endX, endY, startX) in boxes:
                    if (endX-startX) > miniSize and (endY-startY) > miniSize:
                        face_locations.append((startY, endX, endY, startX))

    return face_locations