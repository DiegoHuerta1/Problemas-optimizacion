# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:41:30 2024

@author: diego
"""

import networkx as nx
import matplotlib.pyplot as plt
import sys



# Funcion usada para leer el archivo
# devuelve el grafo creado 
# se le debe de pasar el path del arhcivo
def ReadDataFile(file_instancia):
    
    # comunicar donde se lee la instancia
    print(f"Se lee el grafo del archivo: {file_instancia}")
    
    # grafo vacio
    G = nx.Graph()
    
    # intentar leer el archivo
    try:
        
        # abrirlo
        with open(file_instancia, 'r') as file:
            
            # leer linea por linea
            for line in file:
                
                # cada linea representa una arista
                # la linea: i j
                # representa la arista {i,j} entre los nodos i,j
                
                # separar la linea en i y j
                i, j = line.split()
                
                # agrear la arista correspondiente
                # como numero no como string
                G.add_edge(int(i), int(j))
        return G
    
    # si no se puede leer porque no se encontro
    except FileNotFoundError:
        print("El archivo no fue encontrado")
        sys.exit(1)
        
    
# Dado un dominio discreto X
# y una funcion, representada como un diccionario de pares x:f(x)
# encontrar la x en X que alcanza el valor minimo de f(x)
# Se usa para encontrar el vertice con menor valor de g,
# Es decir, el dominio son los vertices y la funcion es g
def argmin(FeasibleSet, g):
    
    
    # guardar el elemento con menor valor hasta el momento
    k_star = None
    
    # guardar el valor para ese elemento
    minvalue = float('inf')
    
    
    # iterar en los elementos del dominio
    for v in FeasibleSet:
        
        # ver si su valor es menor que el menor actual
        if g[v] < minvalue:
            
            # si es menor, actualizar
            # pues se ha encontrado el nuevo menor
            minvalue = g[v]
            k_star = v
            
    # devolver el elemento que alcanza el menor valor
    return k_star



# actualiza la region de soluciones factibles
# despues de obtener agregar un elemento a la solucion parcial
def UpdateFeasibleSet(G, FeasibleSet, i_star):
    # toma el grafo, la region factible actual
    # y el elemento que se acaba de meter a la solucion parcial
    
    
    # se deben de quitar el elemento seleccionado i_star
    # asi como todos sus vecinos
    
    # primero quitar i_star
    FeasibleSet.remove(i_star)
    
    # tomar los vecinos de i_star
    vecinos_i_star = set(G.neighbors(i_star))
    
    # iterar en los vecinos
    for vecino in vecinos_i_star:
        
        
        # ver si esta en la solucion factible
        if vecino in FeasibleSet:
            
            # si es asi, quitarlo
            FeasibleSet.remove(vecino)
        
    # ya se termino de actualizar
    # devolverla
    return FeasibleSet


# en cada iteracion del algoritmo, se actualiza la funcion voraz
# se implementa como un diccionario de pares llave-valor v:g(v)
def UpdateGreedyFunction(G, FeasibleSet):
    
    # Inicializar el diccionario de la funcion
    g = {}
    
    # iterar en cada vertice del conjunto de elementos factibles
    # para cada uno, poner su evaluacion en la funcion
    for i in FeasibleSet:
        
        # obtener sus vecinos
        vecinos = set(G.neighbors(i))
        
        # su evaluacion en la funcion es la cardinalidad
        # de la interseccion de los vecinos
        # con el conjunto de elementos factibles
        g[i] = len(vecinos.intersection(FeasibleSet))
        
    # devolver la funcion
    return g


# Dibjua una iteracion del algoirtmo voraz
# toma el grafo, la solucion parcial
# el conjutno de elementos factibles
# el pos para ordenar los nodos
# y el numero de iteracion
def Draw_partial_solution(G, S, FeasibleSet, pos, it):  
    
    
    # Delimitar el color de cada nodo
    # verde - nodos en solucion parcial
    # lighgreen - nodos en conjutno de elementos factibles
    # lightgray - nodos que no se pueden escojer
    dict_colores = {"Solucion parcial": "green",
                    "Elementos factibles": "lightgreen",
                    "Nodos bloqueados": "lightgray"}
    
    
    # para cada nodo, ver su color
    node_colors = []
    # iterar en los nodos
    for v in G.nodes():
        
        # si esta en la solucion parcial
        if v in S:
            node_colors.append(dict_colores["Solucion parcial"])
            
        # si esta en el conjunto facible
        elif v in FeasibleSet:
            node_colors.append(dict_colores["Elementos factibles"])
        
        # si no se cumple ninguna, es un nodo bloqueado
        else:
            node_colors.append(dict_colores["Nodos bloqueados"])
            
            
    # dibujar el grafo con el pos y los colores
    nx.draw(G, pos=pos,
            with_labels=True,
            node_color=node_colors,
            width=2.0)
    
    # indicar la iteracion en el titulo
    plt.title("Iteración: "+str(it))
    
    
    # Poner una lenda que indique los colores
    legend_labels = []
    
    # iterar en los colores del diccionario
    for label, color in dict_colores.items():
        
        # agregar su leyenda
        legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10))
    
    # Agrega la leyenda al gráfico
    plt.legend(handles=legend_labels, loc='best')
        
    # mostrar
    plt.show()
    
    

# Dibujar un conjunto independiente maximo
# se usara para mostrar el encontrado con el algoritmo
def Draw_MaxIndSet(G, max_ind_s, pos):
    
    
    
    # Dibujar el grafo
    nx.draw(G, pos=pos, 
            with_labels=True,
            node_color = 'lightblue')
    
    
    # poner los nodos del maximum independent set
    nx.draw_networkx_nodes(max_ind_s, pos=pos,
                           node_color = "red")
    
    
    # añaidr para la leyenda la indicacion de los nodos
    # del maximum independent set
    legend_label = plt.Line2D([0], [0], marker='o', color='w', label="Vertex in Maximum Independet Set", markerfacecolor="red", markersize=10)
    
    # poner la leyenda
    plt.legend(handles= [legend_label], loc='best')
    
    # poner titulo y mostrar
    plt.title("Maximum Independent Set", fontsize=20)
    plt.show()


# Funcion que ejecuta un algoritmo greedy
# para resolver el problema de
# Maximum Independent Set
def Max_Ind_Set(file_instancia):
    
    # se le pasa el nombre y path del arhivo
    # que tiene la instancia del problema
    
    # leer el grafo de esta instancia
    G = ReadDataFile(file_instancia)
    
    # obtener una posicion para dibujar este grafo
    # sera utilizada durante todas las visualizaciones
    pos = nx.kamada_kawai_layout(G)
    
    # Inicia el algoritmo
    
    # Sea S el conjunto de vertices en el conjunto independiente
    # Esto es, la solucion que se va construyendo, inica vacia
    S = set()
    
    # El valor optimizado de la funcion objetivo comienza con 0
    # Recordar que la funcion es la cardinalidad del conjunto independiente
    f = 0
    
    # El conjunto factible comienza siendo igual a todos los nodos
    # Todos los nodos son candidatos para la solucion
    FeasibleSet = set(G.nodes())
    
    # La funcion voraz g comienza siendo el grado de cada nodo
    # Implementar esta funcion como un diccionario
    # Donde las llaves son los nodos v, y sus valores son g(v)
    g = {v: G.degree(v) for v in FeasibleSet}
    
    # Comenzar las iteraciones del algoritmo
    iter_ = 0
    while len(FeasibleSet) > 0:
        
        # indicar que se hizo una iteracion mas
        iter_ += 1
        
        # se escoje el vertice que "mejor" en esta iteracion
        # esto es, el que tenga menor valor en la funcion g
        i_star =  argmin(FeasibleSet, g)
        
        # decir que se selecciono este elemento en la iteracion actual
        print(f"En la iteración {iter_} se selecciona el vertice {i_star}")
        
        # agregar ese vertice a la solucion que se construye
        S.add(i_star)
        
        
        # sumar en uno la funcion, es el valor que se tiene hasta ahora
        # se suma uno poruqe se agrega un vertice a la solucion
        f += 1
        
        # actualizar el conjunto de elementos factibles
        # pues se tienen que quitar el nodo que se acaba de seleccionar
        # asi como todos sus vecinos
        FeasibleSet = UpdateFeasibleSet(G, FeasibleSet, i_star)
        
        # actualizar la funcion g
        # para solo considerar vecinos en el conjunto de elementos factibles
        g = UpdateGreedyFunction(G, FeasibleSet)
        
        
        # dibujar la solucion parcial hasta ahora
        Draw_partial_solution(G, S, FeasibleSet, pos, iter_)
        
        
    # ya se termino de ejecutar el algoritmo
    
    # comprobar que el costo sea el numero de vertices en la solucion
    assert len(S) == f
    
    # Comunicar el costo total alcanzado
    print("El costo total es:", f)
    
    # mostrar graficamente la solucion
    Draw_MaxIndSet(G, S, pos)



# main --------------------------------------------------------------

# ejecutar el algoritmo
Max_Ind_Set("Grafica.txt")
