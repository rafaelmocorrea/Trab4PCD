import enum
import cv2
import numpy as np
import os
import time
from multiprocessing import Lock, Pool, Value
from functools import partial

'''
Metodo 1, usa a area dos contornos
'''

def multi_sumarizador(path, limiar, id, n):
    print(f'Processo {id} iniciou a sumarizacao')
    bg = cv2.createBackgroundSubtractorMOG2()
    video = cv2.VideoCapture(path)
    max_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)/n)
    inicio = int((video.get(cv2.CAP_PROP_FRAME_COUNT)/n)*id)
    fim = inicio + max_frames - 1
    if (id == n - 1):
        max_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) - id*(video.get(cv2.CAP_PROP_FRAME_COUNT)/n))
    video.set(cv2.CAP_PROP_POS_FRAMES,inicio)
    
    area_contornos = []
    frames_conteudo = []
    num_frames_conteudo = []

    conta_frame = 0
    while(conta_frame < max_frames):
        ret,frame = video.read()

        if ret == False:
            print("Acabou")
            break
        
        conta_frame += 1
        mask = bg.apply(frame)
        _,mask = cv2.threshold(mask,150,255,cv2.THRESH_BINARY)
        knl = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        mask = cv2.erode(mask,knl,iterations=2)
        mask = cv2.dilate(mask,knl,iterations=2)
        contornos,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas_aux = []
        if len(contornos) > 0:
            for a in contornos:
                areas_aux.append(cv2.contourArea(a))
            areas_aux = np.array(areas_aux)
            area_contornos.append(areas_aux.mean()) #media das areas do frame
            frames_conteudo.append(mask)
            num_frames_conteudo.append(conta_frame+inicio)
    area_contornos = [n for n in area_contornos if n != 0]
    area_contornos = np.array(area_contornos)    
    if len(num_frames_conteudo) > 0:
        print(f'Processo {id} Filtrando Frames')
        filtra_frames(area_contornos.mean(),frames_conteudo,num_frames_conteudo,limiar)
    video.release()
    num_frames_conteudo = [a for a in num_frames_conteudo if a > -1]
    frames_conteudo = [b for b in frames_conteudo if len(b)>0]
    return num_frames_conteudo, conta_frame
    


def divide_frames(frames, n_procs):
    ret = []
    inicio = 0
    fim = int(frames/n_procs) - 1
    for i in range(n_procs):
        if (i == n_procs - 1):
            ret.append([inicio,int(frames-1)])
        else:
            ret.append([inicio,fim])
            inicio = fim + 1
            fim += int(frames/n_procs)
    print(ret)
    return ret

def divide_video(divisoes, path, fps,n):
    video = cv2.VideoCapture(path)
    conta_frame = 0
    ret,frame = video.read()
    conta_frame = 0
    width = int(video.get(3))
    height = int(video.get(4))
    tam = (width,height)
    outputs = [cv2.VideoWriter(str(i)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,tam) for i in range(n)]
    while ret:
        conta_frame += 1
        for i, divisao in enumerate(divisoes):
            inicio,fim = divisao
            if inicio <= conta_frame <= fim:
                outputs[i].write(frame)
        ret,frame = video.read()
    
    for o in outputs:
        o.release()
    video.release()

def dinamiza(vet,fps,t,limite):
    frames = fps*t
    ret = []
    for i in vet:
        for a in range(i-frames,i+frames+1):
            if ((a not in vet) or a == i) and (a >= 0 and a <= limite and a not in ret):
                ret.append(a)
    return ret

def filename(nome_arq):
    for i in range(len(nome_arq)):
        if nome_arq[-1 - i] == '.':
            ind = len(nome_arq) + (-1-i)
            break
    return nome_arq[0:ind] + '_sum.mp4'

def filtra_frames(media, frames, indices,limiar): #filtra os frames de acordo a média do área dos frames
    for i in range(len(indices)):
        area = 0
        contornos,_ = cv2.findContours(frames[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_array = []
        for a in contornos:
            max_array.append(cv2.contourArea(a))
        max_array = np.array(max_array)
        if max_array.max() < media*(limiar/100): #se não houver um objeto maior que a média ponderada com o limiar o frame é removido
            frames[i] = []
            indices[i] = -1

def sumarizador_frames(path,limiar=20): #retorna o índice e máscaras dos frames de interesse
    print("Sumarizador iniciado")
    bg = cv2.createBackgroundSubtractorMOG2()
    video = cv2.VideoCapture(path)
    conta_frame = 0

    area_contornos = []
    frames_conteudo = []
    num_frames_conteudo = []

    while(1):
        ret,frame = video.read()

        if ret == False:
            print("Acabou")
            break
        
        conta_frame += 1
        mask = bg.apply(frame)
        _,mask = cv2.threshold(mask,150,255,cv2.THRESH_BINARY)
        knl = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        mask = cv2.erode(mask,knl,iterations=2)
        mask = cv2.dilate(mask,knl,iterations=2)
        contornos,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas_aux = []
        if len(contornos) > 0:
            for a in contornos:
                areas_aux.append(cv2.contourArea(a))
            areas_aux = np.array(areas_aux)
            area_contornos.append(areas_aux.mean()) #media das areas do frame
            frames_conteudo.append(mask)
            num_frames_conteudo.append(conta_frame)
    area_contornos = [n for n in area_contornos if n != 0]
    area_contornos = np.array(area_contornos)

    if len(num_frames_conteudo) > 0:
        print("Filtrando Frames")
        filtra_frames(area_contornos.mean(),frames_conteudo,num_frames_conteudo,limiar)
    video.release()
    num_frames_conteudo = [a for a in num_frames_conteudo if a > -1]
    frames_conteudo = [b for b in frames_conteudo if len(b)>0]
    return num_frames_conteudo, conta_frame

def aplica_sumarizacao(path, indices,fps):
    video = cv2.VideoCapture(path)
    conta_frame = 0
    width = int(video.get(3))
    height = int(video.get(4))
    tam = (width,height)
    output = cv2.VideoWriter(filename(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,tam)

    while(video.isOpened()):
        ret,frame = video.read()
        if ret == False:
            break
        conta_frame += 1
        if conta_frame in indices:
            output.write(frame)
        
    video.release()
    output.release()
    print(f'Arquivo salvo em {filename(path)}')

def fachada(arq,t,limiar):
    video = cv2.VideoCapture(arq)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    pre_sumarizacao = time.time()
    frames_filtrados,limite = sumarizador_frames(arq,limiar)
    sumarizacao = time.time() - pre_sumarizacao
    pre_dinamizacao = time.time()
    if t != 0:
        frames_dinamizados = dinamiza(frames_filtrados,fps,t,limite)
    else:
        frames_dinamizados = frames_filtrados
    dinamizacao = time.time() - pre_dinamizacao
    pre_aplicacao = time.time()
    aplica_sumarizacao(arq,frames_dinamizados,fps)
    aplicacao = time.time() - pre_aplicacao
    print(f'Tempo para sumarizacao: {sumarizacao}\nTempo para dinamizacao: {dinamizacao}\nTempo para aplicacao: {aplicacao}') 

def fachada_multi(path,t,limiar,n):
    video = cv2.VideoCapture(path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release
    pre_sumarizacao = time.time()
    lista_argumentos = [[path,20,i,n] for i in range(n)]
    p = Pool(n)
    result = p.starmap(multi_sumarizador,lista_argumentos)
    p.close()
    p.join()
    frames_filtrados = []
    limite = 0
    for i in range(n):
        frames_filtrados += result[i][0]
        limite += result[i][1]
    sumarizacao = time.time() - pre_sumarizacao
    pre_dinamizacao = time.time()
    if t != 0:
        frames_dinamizados = dinamiza(frames_filtrados,fps,t,limite)
    else:
        frames_dinamizados = frames_filtrados
    dinamizacao = time.time() - pre_dinamizacao
    pre_aplicacao = time.time()
    aplica_sumarizacao(path,frames_dinamizados,fps)
    aplicacao = time.time() - pre_aplicacao
    print(f'Tempo para sumarizacao: {sumarizacao}\nTempo para dinamizacao: {dinamizacao}\nTempo para aplicacao: {aplicacao}\nNumero de processos: {n}') 


if __name__ == "__main__":
    path = "" #PATH
    n = os.cpu_count()
    fachada_multi(path,0,50,n)