# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # inicializa o próximo ID de objeto exclusivo junto com dois pedidos
        # dicionários usados para acompanhar o mapeamento de um determinado objeto
        # ID para seu centroide e número de quadros consecutivos que ele possui
        # foi marcado como "desaparecido", respectivamente
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # armazena o número máximo de frames consecutivos de um dado
        # objeto pode ser marcado como "desaparecido" até que
        # precisa cancelar o registro do objeto de rastreamento
        self.maxDisappeared = maxDisappeared

        # armazena a distância máxima entre os centroides a serem associados
        # um objeto -- se a distância for maior que esse máximo
        # distância começaremos a marcar o objeto como "desaparecido"
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # ao registrar um objeto usamos o próximo objeto disponível
        # ID para armazenar o centroide
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # para cancelar o registro de um ID de objeto, excluímos o ID do objeto de
        # ambos os nossos respectivos dicionários
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE

    def update(self, rects):
        # verifica se a lista de retângulos da caixa delimitadora de entrada
        # está vazia
        if len(rects) == 0:
            # faz um loop sobre quaisquer objetos rastreados existentes e os marca
            # como desaparecido
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # se atingimos um número máximo de
                # frames onde um determinado objeto foi marcado como
                # ausente, cancele o registro
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # retorne cedo, pois não há centroides ou informações de rastreamento
            # atualizar
            # retorna self.objects
            return self.bbox

        # inicializa uma matriz de centroides de entrada para o quadro atual
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # loop sobre os retângulos da caixa delimitadora
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE

        # se não estivermos rastreando nenhum objeto no momento, pegue a entrada
        # centróides e registre cada um deles
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

        # caso contrário, estão rastreando objetos no momento, então precisamos
        # tenta combinar os centroides de entrada com o objeto existente
        # centróides
        else:
            # pegue o conjunto de IDs de objetos e centroides correspondentes
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # calcula a distância entre cada par de objetos
            # centroides e centroides de entrada, respectivamente -- nosso
            # objetivo será corresponder um centroide de entrada a um existente
            # centroide do objeto
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # para realizar esta correspondência devemos (1) encontrar o
            # menor valor em cada linha e então (2) classificar a linha
            # índices com base em seus valores mínimos para que a linha
            # com o menor valor como na *frente* do índice
            # Lista
            rows = D.min(axis=1).argsort()

            # em seguida, realizamos um processo semelhante nas colunas por
            # encontrando o menor valor em cada coluna e então
            # ordenando usando a lista de índices de linhas previamente computada
            cols = D.argmin(axis=1)[rows]

            # para determinar se precisamos atualizar, registrar,
            # ou cancelar o registro de um objeto que precisamos acompanhar
            # dos índices de linhas e colunas que já examinamos
            usedRows = set()
            usedCols = set()

            # faz um loop sobre a combinação do índice (linha, coluna)
            # tuplas
            for (row, col) in zip(rows, cols):
                # se já examinamos a linha ou
                # valor da coluna antes, ignore-o
                if row in usedRows or col in usedCols:
                    continue

                # se a distância entre os centróides for maior que
                # a distância máxima, não associe os dois
                # centroides para o mesmo objeto
                if D[row, col] > self.maxDistance:
                    continue

                # caso contrário, pegue o ID do objeto para a linha atual,
                # definir seu novo centroide e redefinir o desaparecido
                # contador
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                # indica que examinamos cada uma das linhas e
                # índices de coluna, respectivamente
                usedRows.add(row)
                usedCols.add(col)

            # calcula o índice de linha e coluna que ainda NÃO temos
            # examinado
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # no caso de o número de centroides do objeto ser
            # igual ou maior que o número de centróides de entrada
            # precisamos verificar e ver se alguns desses objetos têm
            # potencialmente desaparecido
            if D.shape[0] >= D.shape[1]:
                # faz um loop sobre os índices de linha não utilizados
                for row in unusedRows:
                    # pega o ID do objeto para a linha correspondente
                    # indexa e incrementa o contador desaparecido
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # verifica se o número de
                    # frames o objeto foi marcado como "desaparecido"
                    # para warrants que cancelam o registro do objeto
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # caso contrário, se o número de centróides de entrada for maior
            # do que o número de centroides de objetos existentes que precisamos
            # registra cada novo centroide de entrada como um objeto rastreável
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # retorna o conjunto de objetos rastreáveis
        # retorna self.objects
        return self.bbox

