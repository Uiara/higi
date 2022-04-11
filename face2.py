from sklearn.feature_extraction import img_to_graph
from app1 import app1
from datetime import datetime
from os import listdir
from os.path import isfile, join
from numpy import *
from PIL import Image, ImageEnhance

import torch
import requests
import face_recognition
import cv2
import numpy as np
import time
import os
import pickle
import tempfile
import numpy
import albumentations

''' Variáveis de ambiente '''
miniSize = 134 #Facial detect parametros
quantidade_fotos_maxima = 5000
face_image = []
frame_number = 0 #variable to count number of frames
face_par = 0.2
requisicao = True
contador = 0

''' Diretórios de arquivos '''
path_origin = "/home/jetson/higienize_jetson/"  
path_dnn = (r"{}/face_detection_model".format(path_origin))
path_send = (r'{}/SEND_LEITO/'.format(path_origin))#path to save face images
path_in = (r'{}/IN_LEITO/'.format(path_origin)) #pasta com as imagens a serem tratadas

print("\n\n\n\n ##########Aplicação inicializada########## \n\n\n\n")

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

while True:
    path_list = os.listdir(path_in)
    if len(path_list) > 0:
        print('\n')
        lista = []
        #cria uma lista com o timestamp de forma ordenada
        for file_ in path_list:
            timestamp = file_.split(".jpg")[0]
            lista.append(timestamp)
            lista.sort()
            quantidade_fotos = len(lista)
        for x in lista:
            img = cv2.imread(str(path_in) + str(x) + '.jpg')
            img_caminho = (str(path_in) + str(x) + '.jpg')
            timestamp_str = str(x)
            frame = numpy.array(img)
            frame_aux = numpy.array(img)
            frame_number += 1 #counting frames

            face_locations = app1.opencv_dnn_face_location(filename = img_caminho, path_dnn_face_detector = path_dnn, miniSize = miniSize)
            detect =  len(face_locations) #quantidade de faces localizada
            arq =  os.listdir(path_send)
            quantidade_temporario = len(arq) #quantas fotos já foram armazenadas
            (h, w) = img.shape[:2]
            rects = []
            for i in np.arange(0, face_locations.shape[2]): # arange, cria um arranjo contendo uma seqüência de valores especificados em um intervalo com início e fim dados
                confidence = face_locations[0, 0, i, 2]
                #if confidence > 0.5: # verdade a distância entre o teste e a imagem mais próxima encontrada.
                    #idx = int(face_locations[0, 0, i, 1])

                    #if CLASSES[idx] != "person":
                        #continue

                person_box = face_locations[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box) # posição de cada ID, formato array
                
                boundingboxes = np.array(rects) # caixa vermelha
                boundingboxes = boundingboxes.astype(int) # caixa vermelha 
                rects = non_max_suppression_fast(boundingboxes, 0.3)

                objects = tracker.update(rects)

'''            
            if detect == 0: #nenhuma face detectada
                contador +=1 
                if requisicao == True and quantidade_temporario == 0 and contador>27:
                    app1.send_request_image(frame_aux=frame_aux, path_send=path_send, home=path_origin)
                    requisicao = False
                    print(requisicao)
            else:
                contador = 0
                requisicao = True
'''            
            for (top, right, bottom, left) in face_locations: #limita a 5000 imagens salvas
                if quantidade_temporario <= quantidade_fotos_maxima:
                    print("\nImagem salva")
                    app1.save_detected_face(timestamp_str,face_par, top,bottom,left,right,frame_aux, detect, path_send=path_send, home=path_origin) #salva a face detectada
            print("Frame de número: {}".format(frame_number), "\nFaces: {}".format(detect))
            time.sleep(0.5)
            os.remove(img_caminho)
            print('\n')
    for (objectId, bbox) in objects.items():
    #inicio = time.time()
    print(inicio)

    x1, y1, x2, y2 = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    fim = time.time()
    #print(fim - inicio)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # desenhar um retângulo em qualquer imagem
    text = "ID: {}".format(objectId)
    cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1) # desenhar uma string de texto em qualquer imagem
    #print(str(text))
    fps_end_time = datetime.datetime.now()
    #time_diff = fps_end_time - fps_start_time
    #if time_diff.seconds == 0:
    #    fps = 0.0
    #else:
    #    fps = (total_frames / time_diff.seconds)
    
    fps_text = "FPS: {:.2f}".format(fps)
    print(str(time_diff)+" _ "+str(fps_text)+" _ ")

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    else:
        print('Não há fotos na pasta')
        time.sleep(0.5)