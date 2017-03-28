#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:11:36 2017

@author: francisco
"""

##########################################################################
#####  Sistema de recomendación aplicado a Quora Question Pairs
#####      PLNCD(II)
#####          Juan José Martín Guareño
##########################################################################

##########################################################################
### Paso 1: Leer ficheros train y test.
##########################################################################

import csv

import numpy as np
import pandas as pd
from pandas import DataFrame

from time import time
  
#DF=DataFrame(train,columns=['id','qid1','qid2','Cuestion1','Cuestion2','es_Duplicado'])
#DF

# Para crear un df del tamaño que quiera:
 
#i_max=20
#with open('Quora Question Pairs/train.csv') as csvarchivo:
#    archivo = csv.reader(csvarchivo)
#    ids=[]
#    qid1=[]
#    qid2=[]
#    c1=[]
#    c2=[]
#    es_D=[]
#    i=0
#    for linea in archivo:
#        while i < i_max:
#            ids.append(linea[0])
#            qid1.append(linea[1])
#            qid2.append(linea[2])
#            c1.append(linea[3])
#            c2.append(linea[4])
#            es_D.append(linea[5])
#            i+=1
#            print(i)
#            break
#    train20 = {'id' : ids,
#          'qid1' : qid1,
#          'qid2' : qid2,
#          'Cuestion1' : c1,
#          'Cuestion2' : c2,
#          'es_Duplicado' : es_D}
#    train20df = pd.DataFrame(train20)
    
    
#Creamos los diccionarios como nos interesa y el numero que queramos
 
def leeCuestiones2(i_max=100000000):
    with open('Quora Question Pairs/train.csv') as csvarchivo:
        archivo = csv.reader(csvarchivo)
        tr={}
        i=-1
        for linea in archivo:
            while i < i_max:
                if i != -1:
                    tr[i] = {'id' :linea[0],
                          'qid1' : linea[1],
                          'qid2' : linea[2],
                          'Cuestion1' : linea[3],
                          'Cuestion2' : linea[4],
                          'es_Duplicado' : linea[5]}
                i+=1
                break
    return(tr)
    
def leeCuestiones(i_max=100000000): #Para que por defecto las lea todas

    with open('Quora Question Pairs/train.csv') as csvarchivo:
        archivo = csv.reader(csvarchivo)
        Cuestiones={}
        i=-1
        for linea in archivo:
            while i < i_max:
                dic1={}
                dic2={}
                if i != -1:
                    dic1['Cuestion']=linea[3]
                    dic2['Cuestion']=linea[4]
                    dic1['id']=linea[1]
                    dic2['id']=linea[2]
                    
                    Cuestiones[linea[1]] = dic1
                    Cuestiones[linea[2]] = dic2
                i+=1
                break   
    return Cuestiones

    
#with open('Quora Question Pairs/train.csv') as csvarchivo:
#    archivo = csv.reader(csvarchivo)
#    ids=[]
#    qid1=[]
#    qid2=[]
#    c1=[]
#    c2=[]
#    es_D=[]
#    for linea in archivo:
#        ids.append(linea[0])
#        qid1.append(linea[1])
#        qid2.append(linea[2])
#        c1.append(linea[3])
#        c2.append(linea[4])
#        es_D.append(linea[5])
#    df = {'id' : ids,
#          'qid1' : qid1,
#          'qid2' : qid2,
#          'Cuestion 1' : c1,
#          'Cuestion 2' : c2,
#          'es_Duplicado' : es_D}
#    train2 = pd.DataFrame(df)
#    
## Asi es ,ás sencillo

#train = pd.read_csv('Quora Question Pairs/train.csv', index_col='id')

### Hasta aquí tenemos dos formas de obtenrt un df con los datos train que disponemos

##########################################################################
### Paso 2: Preprocesado y limpieza de los resúmenes de las películas
##########################################################################

import nltk

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

#valen con spanish para español pero lo queremos en ingles
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

def obtenerNombresPropios(nombres, texto): # Va regular
    # Recorremos todas las oraciones de un texto (resumen de una película)

    for frase in nltk.sent_tokenize(texto):
        #
        # nltk.word_tokenize devuelve la lista de palabras que forman
        #    la frase (tokenización)
        #
        # nltk.pos_tag devuelve el part of speech (categoría) correspondiente
        #    a la palabra introducida
        #
        # nltk.ne_chunk devuelve la etiqueta correspondiente al part of
        #    speech
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(frase))):
            try:
                if chunk.label() == 'PERSON':
                    for c in chunk.leaves():
                        if str(c[0].lower()) not in nombres:
                            nombres.append(str(c[0]).lower())
            except AttributeError:
                pass
    return nombres

def preprocesarPreguntas(Preguntas):
    print("Preprocesando Preguntas")
    nombresPropios = []

    for elemento in Preguntas:
        print("Preproceso cuestión: ",elemento)

        Pregunta = Preguntas[elemento]

        ## Eliminación de signos de puntuación usando tokenizer
        cuestion = Pregunta['Cuestion']
        #texto = ' '.join(tokenizer.tokenize(resumen))
        texto = ' '.join(tokenizer.tokenize(cuestion))
        Pregunta['texto'] = texto

        nombresPropios = obtenerNombresPropios(nombresPropios, texto)

    ignoraPalabras = stopWords
    #ignoraPalabras.union(nombresPropios)

    palabras = [[]]
    for elemento in Preguntas:
        Pregunta = Preguntas[elemento]

        texto = Pregunta['texto']
        textoPreprocesado = []
        for palabra in tokenizer.tokenize(texto):
            
            if (palabra.lower() not in ignoraPalabras):
                textoPreprocesado.append(stemmer.stem(palabra.lower()))
                palabras.append([(stemmer.stem(palabra.lower()))])

        Pregunta['texto'] = ' '.join(textoPreprocesado)
        
    return palabras


##########################################################################
### Paso 3: Creación de la colección de textos
##########################################################################

from gensim import corpora, models, similarities
    
def crearColeccionTextos(Preguntas):
    print("Creando colección global de resúmenes")
    textos = []
    
    for elemento in Preguntas:
        Pregunta = Preguntas[elemento]
        texto = Pregunta['texto']
        lista = texto.split(' ')

        textos.append(lista)

    return textos

##########################################################################
### Paso 4: Creación del diccionario de palabras
##########################################################################
###
### El diccionario está formado por la concatenación de todas las
### palabras que aparecen en alguna sinopsis (modo texto) de alguna
### de las peliculas
###
### Básicamente esta función mapea cada palabra única con su identificador
###
### Es decir, si tenemos N palabras, lo que conseguiremos al final
### es que cada película sea representada mediante un vector en un
### espacio de N dimensiones

def crearDiccionario(textos):
    print("Creación del diccionario global")
    return corpora.Dictionary(textos)

## diccionario con las palabras y sus indices correspondientes
def CreaDic_pal(lista_palabras):
    dic_pal=dict()
    for i in range(1,len(lista_palabras)):
        dic_pal[lista_palabras[i][0]]=i
    return dic_pal

##########################################################################
### Paso 5: Creación del corpus de resúmenes preprocesados
##########################################################################
###
### Crearemos un corpus con la colección de todos los resúmenes
### previamente pre-procesados y transformados usando el diccionario
###

def crearCorpus(textos,diccionario):
    print("Creación del corpus global con los resúmenes de todas las preguntas")
    
    return [diccionario.doc2bow(texto) for texto in textos]


## nos creamos otro corpus donde entren los sinonimos de cada palabra
from nltk.corpus import wordnet
def sin_ant(palabra):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(palabra):
        for l in syn.lemmas():
            synonyms.append(l.name()) 
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return(set(synonyms))
    #return(set(antonyms))



#def CreaCorpus_sin(textos,diccionario,palabras,dic_pal,corpus):
    
    corp_mod=[]
    for lis_par in corpus:
        corp_mod_lis=[]
        for (a,b) in lis_par:
            sin=sin_ant(palabras[(a+1)][0])
            long=len(sin)
            corp_mod_lis.append((a,b))
            for a1 in sin:
                if a1 in dic_pal.keys():
                    corp_mod_lis.append((dic_pal[a1],b/long))
                else:
                    palabras.append(a1)
                    l=len(palabras)
                    dic_pal[a1]=l
                    corp_mod_lis.append((l,b/long))
        corp_mod.append(corp_mod_lis)
    return [corp_mod,palabras,dic_pal]


##########################################################################
### Paso 6: Creación del modelo tf-idf
##########################################################################

def crearTfIdf(corpus):
    print("Creación del Modelo Espacio-Vector Tf-Idf")
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

##########################################################################
### Paso 7: Creación del modelo LSA (Latent Semantic Analysis)
##########################################################################

#import gensim
#import numpy as np

### Valores clave para controlar el proceso
#TOTAL_TOPICOS_LSA = 20
#UMBRAL_SIMILITUD = 0.98

def crearLSA(corpus,pel_tfidf,diccionario,TOTAL_TOPICOS_LSA,UMBRAL_SIMILITUD):
    print("Creación del modelo LSA: Latent Semantic Analysis")
    #numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms = 50000)
    #svd = np.linalg.svd(numpy_matrix, full_matrices=False, compute_uv = False)

    lsi = models.LsiModel(pel_tfidf, id2word=diccionario, num_topics=TOTAL_TOPICOS_LSA)

    indice = similarities.MatrixSimilarity(lsi[pel_tfidf]) ## Te devuelve una lista de listas por cada pregunta su relacion con las demas cluster

    return (lsi,indice)



def crearCodigosPreguntas(Preguntas):
    codigosPreguntas = []
    for i, elemento in enumerate(Preguntas):
        Pregunta = Preguntas[elemento]
        codigosPreguntas.append(Pregunta['id'])
    return codigosPreguntas



def crearModeloSimilitud(Preguntas, pel_tfidf,lsi,indice,UMBRAL_SIMILITUD=0.95 ,salida=None):
    codigosPreguntas = crearCodigosPreguntas(Preguntas)
    print("Creando enlaces de similitud entre películas")
    if (salida != None):
        print("Generando salida en fichero ",salida)
        ficheroSalida = open(salida,"w")
        
    for i, doc in enumerate(pel_tfidf):
        print("============================")
        PreguntaI = Preguntas[codigosPreguntas[i]]
        
        print("Pregunta I = ",PreguntaI['id'],"  " ,PreguntaI['Cuestion'])

        if (salida != None):
            ficheroSalida.write("============================")
            ficheroSalida.write("\n")
            ficheroSalida.write("Pregunta I = " + PreguntaI['id'] + "  " + PreguntaI['Cuestion'])
            ficheroSalida.write("\n")
            
        vec_lsi = lsi[doc]
        #print(vec_lsi)
        indice_similitud = indice[vec_lsi]
        similares = []
        for j, elemento in enumerate(Preguntas):
            s = indice_similitud[j]
            if (s > UMBRAL_SIMILITUD) & (i != j):
                PreguntaJ = Preguntas[codigosPreguntas[j]]
                similares.append((codigosPreguntas[j], s))
                
                print("   Similitud: ",s,"   ==> Pregunta J = ",PreguntaJ['id'],"  ",PreguntaJ['Cuestion'])
                if (salida != None):
                    ficheroSalida.write("   Similitud: " + str(s) + "   ==> Pregunta J = " + PreguntaJ['id'] + "  " + PreguntaJ['Cuestion'])
                    ficheroSalida.write("\n")
                    
            similares = sorted(similares, key=lambda item: -item[1])

            PreguntaI['similares'] = similares
            PreguntaI['totalSimilares'] = len(similares)

    if (salida != None):
        ficheroSalida.close()


#Cuestiones   = leeCuestiones(20)
###Esto es mi train
#palabras    = preprocesarPreguntas(Cuestiones)
#textos      = crearColeccionTextos(Cuestiones)
#diccionario = crearDiccionario(textos)
#corpus      = crearCorpus(diccionario)
##
#pre_tfidf   = crearTfIdf(corpus)
#(lsi,indice)= crearLSA(corpus,pre_tfidf)
##crearModeloSimilitud(Cuestiones,pre_tfidf,lsi,indice,"cuestiones1Similares.txt")
#crearModeloSimilitud(Cuestiones,pre_tfidf,lsi,indice)


















