#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:38:32 2017

@author: francisco
"""


from Ejercicio_SR import *

from time import time
import pandas as pd
import io
import json

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

def Prueba(TOTAL_TOPICOS_LSA=20,UMBRAL_SIMILITUD=0.95,i_max=100,salida=None):#Por defecto 100
    dic_Acceso={}
    Cuestiones   = leeCuestiones(i_max)
    palabras    = preprocesarPreguntas(Cuestiones)
    textos      = crearColeccionTextos(Cuestiones)
    diccionario = crearDiccionario(textos)
    dic_pal     = CreaDic_pal(palabras)
    corpus      = crearCorpus(textos,diccionario)
    corpus_sin  = CreaCorpus_sin(textos,diccionario,palabras,dic_pal,corpus)

    pre_tfidf   = crearTfIdf(corpus_sin)
    (lsi,indice)= crearLSA(corpus_sin,pre_tfidf,diccionario,TOTAL_TOPICOS_LSA,UMBRAL_SIMILITUD)
#    if salida:
#        cr=crearModeloSimilitud(Cuestiones,pre_tfidf,lsi,indice,UMBRAL_SIMILITUD,salida)
#    else:
#        cr=crearModeloSimilitud(Cuestiones,pre_tfidf,lsi,indice,UMBRAL_SIMILITUD)
    
    dic_Acceso['Cuestiones']=Cuestiones  
    dic_Acceso['palabras']=palabras
    dic_Acceso['textos']=textos
    dic_Acceso['diccionario']=diccionario
    dic_Acceso['corpus']= corpus 
    dic_Acceso['corpus_sin']= corpus_sin 
    dic_Acceso['pre_tfidf']=pre_tfidf   
    #dic_Acceso['(lsi,indice)']=(lsi,indice)
    #dic_Acceso['Modelos']=cr
    dic_Acceso['dic_pal']=dic_pal
    
    return(dic_Acceso)
    
    

def similares(st,Cues):
    sim=[]
    for (a,b) in Cues[st]['similares']:
        sim.append(a)
    return(sim)

def is_duplicate(qid1,qid2,Cues):
    st1=str(qid1)
    st2=str(qid2)
    return((st1 in similares(st2,Cues)) or (st2 in similares(st1,Cues)))

def CreaCorpus_sin(textos,diccionario,palabras,dic_pal,corpus):
    
    corp_mod=[]
    for lis_par in corpus:
        corp_mod_lis=[]
      #  lista_pal=[]
        for (a,b) in lis_par:
            sin=sin_ant(palabras[(a+1)][0])
            long=len(sin)
            corp_mod_lis.append((a,b))
            #lista_pal.append(a)
            for a1 in sin:
                if not a1==a:
                    if a1 in dic_pal.keys():
                        corp_mod_lis.append((dic_pal[a1],b/long))
                         #   lista_pal.append(a1)
                    else:
                        pass
                   # else:
                    #    if a1 in dic_pal.keys():
                     #       nuevo_b=b/long + 
                      #      corp_mod_lis.append((dic_pal[a1],b/long))
                        
                 
        corp_mod.append(corp_mod_lis)
    sol=[]
    for par in corp_mod:
        sol.append(sorted(par))
        
    return (sol)

def Tasa_acierto(TOTAL_TOPICOS_LSA=50,UMBRAL_SIMILITUD=0.95,i_max=10000000,Cues=None): ## Analizado que lo hace bien
    
    t1=time()
    if not Cues:
        Cues=Prueba(TOTAL_TOPICOS_LSA,UMBRAL_SIMILITUD,i_max)['Cuestiones']
    C=Cues    
    tr=leeCuestiones2(i_max)
    Acierto=0
    Falsos_Positivos=0
    Falsos_Negativos=0
    
    long=len(tr)
    for i in range(long):
        par=tr[i]
        is_d_V=is_duplicate(par['qid1'],par['qid2'],C)
        #print(is_d_V)
        is_d=bool(int(par['es_Duplicado']))
        #print(is_d)
        if (is_d_V==is_d):
            #print("Acierta\n")
            Acierto+=1
        elif (is_d_V > is_d):
            Falsos_Positivos+=1
        else:
            Falsos_Negativos+=1
        #else:
            #print("Falla\n")
    
    print("Tasa de acierto  :",(Acierto/long)*100,"%")
    print("Falsos Positivos :",(Falsos_Positivos/long)*100,"%")
    print("Falsos Negativos :",(Falsos_Negativos/long)*100,"%")
    t2=time()
    print("Tiempo :",t2-t1,"s\n")
    return [Acierto/long,Falsos_Positivos/long,Falsos_Negativos/long]





def seleccion(n,umbrales=None,grupos=None):
    tiempo=time()
    Umbrales=[0.5,0.75,0.9,0.99,0.99999]+umbrales
    u=Umbrales
    Grupos=[round((n/10),0),round((n/5),0),round((n/2),0),n]+grupos
    g=Grupos
    
    dic={}
    dic_e1={}
    dic_e2={}
    for i in u:
        dic_j={}
        dic_j_e1={}
        dic_j_e2={}
        for j in g:
            l=Tasa_acierto(j,i,n)
            dic_j[j]=l[0]
            dic_j_e1[j]=l[1]
            dic_j_e2[j]=l[2]
        dic[i]=dic_j
        dic_e1[i]=dic_j_e1
        dic_e2[i]=dic_j_e2
    Df=pd.DataFrame(dic)
    Df_e1=pd.DataFrame(dic_e1)
    Df_e2=pd.DataFrame(dic_e2)
    print("Con n=",n)
    print(Df_e1)
    print(Df_e2)
    tf=time()
    print("El tiempo total ha sido:",(tf-tiempo),"s")
    Dic_Guardado={}
    Dic_Guardado["Acierto"]=dic
    Dic_Guardado["Falsos Positivos"]=dic_e1
    Dic_Guardado["Falsos Negativos"]=dic_e2
    Dic_Guardado["Tiempo de ejecución"]=("El tiempo total ha sido:",round(tf-tiempo),"s")
    
    with io.open('/Users/francisco/Documents/Master DS y BD/Segundo Cuatrimestre/PLN(II)/Pruebas/ n_'+str(n)+'.json', 'w', encoding='utf-8') as f:
           f.write((json.dumps(Dic_Guardado, ensure_ascii=False)))

    return(Df)
            




Cuestiones   = leeCuestiones(20)
palabras    = preprocesarPreguntas(Cuestiones)
textos      = crearColeccionTextos(Cuestiones)
diccionario = crearDiccionario(textos)
dic_pal     = CreaDic_pal(palabras)
corpus      = crearCorpus(textos,diccionario)
corpus_sin  = CreaCorpus_sin(textos,diccionario,palabras,dic_pal,corpus)

pre_tfidf   = crearTfIdf(corpus_sin)
(lsi,indice)= crearLSA(corpus_sin,pre_tfidf,diccionario,20,0.95)
if salida:
    cr=crearModeloSimilitud(Cuestiones,pre_tfidf,lsi,indice,UMBRAL_SIMILITUD,salida)
else:
    cr=crearModeloSimilitud(Cuestiones,pre_tfidf,lsi,indice,UMBRAL_SIMILITUD)
                

