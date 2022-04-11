import cv2
import datetime
import imutils
import time
import numpy
from matplotlib import image
import numpy as np
from app1 import app1
from centroidtracker import CentroidTracker

miniSize = 134
path_send = '/home/bianka/projet_higi_0.2/SEND/'

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
#protoPath = os.path.sep.join([path_dnn_face_detector, "deploy.prototxt"])
#modelPath = os.path.sep.join([path_dnn_face_detector, "res10_300x300_ssd_iter_140000.caffemodel"])

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

face_par =0.2

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i": #dtype.kind identifica o tipo do caracter
            boxes = boxes.astype("float") #astype cópia da matriz, convertida para um tipo especificado.

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2) #argsort retorna os índices que classificariam um array.

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            '''????'''
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Ocorreu uma exceção em non_max_suppression : {}".format(e))



def main():
    total_frames = 0
    cap = cv2.VideoCapture('test_video.mp4')
    face_locations = []
    while True:
        ret, frame = cap.read()
        cv2.imwrite('aux.jpg',frame)
        imagem = 'aux.jpg'

        frame_aux = numpy.array(frame)

        frame = imutils.resize(frame, width=600) # redimensona a imagem
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2] # retorna uma tupla do número de linhas, colunas e canais

        #blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5) # aprendizado profundo
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        detector.setInput(blob) # define o novo valor para o blob de saída da camada
        person_detections = detector.forward() # executa uma passagem para frente para calcular a saída líquida.
        rects = []
        boxes = []
        for i in np.arange(0, person_detections.shape[2]): # arange, cria um arranjo contendo uma seqüência de valores especificados em um intervalo com início e fim dados
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5: # verdade a distância entre o teste e a imagem mais próxima encontrada.
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                if (startY <= 1.1*H and endY <= 1.1*H and endX <= 1.1*W and startX <= 1.1*W):
                    #reordering coordenates to use in face recognition
                    boxes.append((int(startY), int(endX), int(endY), int(startX)))
                    for (startY, endX, endY, startX) in boxes:
                        if (endX-startX) > miniSize:
                            if(endY-startY) > miniSize:
                                face_locations.append((startY, endX, endY, startX))
                detect = len(face_locations) #quantidade de faces localizadas

                rects.append(person_box) # posição de cada ID, formato array
        for (top, right, bottom, left) in face_locations:
        # arq =  os.listdir(path_send)
        # quantidade_temporario = len(arq) #quantas fotos já foram armazenadas
            print("salvar imagem")
            app1.save_detected_face(face_par, top,bottom,left,right,frame_aux, detect, path_send) #salva a face detectada
        boundingboxes = np.array(rects) # caixa vermelha
        boundingboxes = boundingboxes.astype(int) # caixa vermelha 
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)

        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # desenhar um retângulo em qualquer imagem
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1) # desenhar uma string de texto em qualquer imagem
            #print(str(text))
        fps_end_time = datetime.datetime.now()
        
        #cv2.putText(frame, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
#    face_locations = app1.opencv_dnn_face_location(filename = "aux.jpg", path_dnn_face_detector = path_dnn, miniSize = miniSize)

