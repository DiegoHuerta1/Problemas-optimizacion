# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:46:39 2024

@author: diego
"""

### LIBRARIES ###
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import matplotlib.ticker as ticker
import seaborn as sns
from itertools import combinations
import pandas as pd

sns.set_theme()

### FUNCTIONS ###



# ------------------------------------------------------------------------

# Funcion para crear el grafo, esto obviamente pone tambien la distancia
# entre los puntos, y define el pos

def CreateGraph(n=10, seed = None):
    """
    Create a complete graph with n nodes randomly located in a 2D space.
    Each edge is assigned its length between its endpoints.
    """
    
    if seed is not None:
        np.random.seed(seed)

    
    points = np.random.rand(n, 2)
    G = nx.complete_graph(len(points))
    for i, (x, y) in enumerate(points):
        G.nodes[i]['pos'] = (x, y)
    for e in G.edges():
        G[e[0]][e[1]]['length'] = np.linalg.norm(points[e[0]]-points[e[1]])
    return G


# ------------------------------------------------------------------------

# Funciones para visualizar


# dibuja una solucion, es decir, un camino
def DrawSolution(G, S, pos, text=''):
    
    """
    Draw the tour defined by a permutation of nodes.
    """
    
    # crear el tour
    tour = CreateTour(S)
    
    # hacer la figura
    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
    
    # dibujar el tour creado
    nx.draw(G, pos, edgelist=tour, ax=ax,
            node_size=70, node_color='darkcyan')
    
    
    # calcular el costo
    cost = np.round(f(G, S), 4)
    
    # titulo de la figura
    ax.set_title(text + f'\nCost: {cost}')
    
    plt.show()
    plt.close()

# ------------------------------------------------------------------------


# solucion inicial aleatoria
def S0_random(G):
    """
    Generate a random initial solution for the TSP.
    """
    n = G.order()
    random.seed(1)
    return random.sample(range(n), n)



# hace un ciclo (en pares) dada una sucecion de nodos
# toma: [1, 2, 3]
# regresa: [(1, 2), (2, 3), (3, 1)]
def CreateTour(S):
    """
    Create a tour from a permutation of nodes.
    """
    tour = [(S[i], S[i+1]) for i in range(len(S)-1)] + [(S[-1], S[0])]
    return tour



# funcion de evaluacion de una solucion
def f(G, S):
    """
     Calculate the total length of a tour  defined by solution.
     """
    tour = CreateTour(S)
    sum_length = sum(G[u][v]['length'] for u, v in tour)
    return sum_length


# ------------------------------------------------------------------------


# funciones de movimientos


# para las funciones N_1


# el movimiento n1
# cambia un numero en la lista con el anterior
def move_n1(S, i):
    """
    Swap the positions of two consecutive indices of a tour.
    """
    S_prime = S.copy()
    S_prime[i-1], S_prime[i] = S[i], S[i-1]
    return S_prime


# vecindario N1
def N1(S):
    """
    Generate neighbors of a solution by applying
    move N1 to each pair of consecutive indices.
    
    Vecindario N1, aplica el movimiento n1 obviamente
    devuelve el movimiento usado para cada vecino
    decir, devuelve una lista, donde cada elemento es: [S_prima, i],
    S_prima es el vecino, y i es el movimiento aplicado a S para obtenerlo,
    es decir, para obtener S_prima se cambia el elemento i-esimo de S por el anterior
    """
    
    # generar todos los posibles vecinos
    neighbor = []
    n = len(S)

    # todos los movimietnos posibles
    for i in range(0, n):
        S_prime = move_n1(S, i)
        # devolver el vecino y el movimiento
        neighbor.append([S_prime, i])
    return neighbor




# obtiene el admissible neighbor de una solucion S
def admisible_N1(S, tabu_list, admisible_treshold, G):
    
    '''
    Obtiene el admissible neighbor de una solucion S en el grafo G,
    dada la lista tabu y el criterio de aspiracion
    Es decir, obtiene todos los vecinos S_prima de S
    tal que, para obtenerlos no se use algun movimiento
    de la lista tabu, o que cumplan el criterio de aspiracion
    esto es, que tengan menor evaluacion en la funcion objetivo
    que el admisible_treshold.
    
    Tambien devuelve los movimientos para encontrar cada uno
    
    '''
    
    #  obtener todo el vecindario completo, saber el movimiento de cada uno
    vecinos_movimiento = N1(S)
    
    # filtrar solo los admisibles
    # todos los vecinos tal que su movimiento no este en la lista tabu
    # o que sean mejores que el admisible_treshold
    adm_n_m = [v_m for v_m in vecinos_movimiento
               if v_m[1] not in tabu_list 
               or f(G, S=v_m[0]) < admisible_treshold]
    
    # devolve los vecinos admisibles con su movimiento
    return adm_n_m



# enocntrar el mejor vecino admisible
def best_admisible_N1(S, tabu_list, admisible_treshold, G):
    
    '''
    Toma una solucion, la lista tabu, y el criterio de aspiracion
    que es un treshold de la mejor solucion encontrada
    
    
    Encuentra el mejor vecino admisible de la soluicon actual
    '''
    
    # tomar todos los vecinos admisibles con sus movimientos
    vecinos_adm_mov = admisible_N1(S, 
                                   tabu_list = tabu_list,
                                   admisible_treshold = admisible_treshold,
                                   G = G)
    
    
    # ahorita que no se conoce cual es el mejor
    # decir que es el primero
    best_n_mov = vecinos_adm_mov[0]
    f_best_n = f(G, S=best_n_mov[0])
    
    
    # iterar en todos los vecinos admisibles
    for vecino_m in vecinos_adm_mov:
        
        # calcular que tan bueno es
        f_vecino = f(G, S=vecino_m[0])
        
        # si es el mejor hasta ahora
        if f_vecino < f_best_n:
            # entonce este es el mejor hasta ahora
            # actualizar
            best_n_mov = vecino_m
            f_best_n = f_vecino
            
            
    # se termina de ver los vecinos
    # se devuelve lo mejor encontrado
    return best_n_mov


# ------------------------------------------------------------------------


# funciones de movimientos


# para las funciones N_2


# el movimiento n2
# cambia el indice de dos numeros
def move_n2(S, i, j):
    """
    Swap the positions of any two indices in a tour
    """
    S_prime = S.copy()
    S_prime[i], S_prime[j] = S[j], S[i]
    return S_prime


# vecindario N2
def N2(S):
    """
    Generate neighbors of a solution by applying
    move N2 to each pair of consecutive indices.
    
    Vecindario N2, aplica el movimiento n2 obviamente
    devuelve el movimiento usado para cada vecino
    decir, devuelve una lista, donde cada elemento es: [S_prima, {i, j}],
    S_prima es el vecino, y {i, j} es el movimiento aplicado a S para obtenerlo,
    es decir, para obtener S_prima se cambian los elementos i y j
    """
    
    # generar todos los posibles vecinos
    n = len(S)
    neighbor = []
    combinations_ = list(combinations(range(n), 2))
    
    

    for i, j in combinations_:
        S_prime = move_n2(S, i, j)
        # devolver vecino y movimiento
        neighbor.append([S_prime, {i, j}])
    return neighbor




# obtiene el admissible neighbor de una solucion S
def admisible_N2(S, tabu_list, admisible_treshold, G):
    
    '''
    Obtiene el admissible neighbor de una solucion S en el grafo G,
    dada la lista tabu y el criterio de aspiracion
    Es decir, obtiene todos los vecinos S_prima de S
    tal que, para obtenerlos no se use algun movimiento
    de la lista tabu, o que cumplan el criterio de aspiracion
    esto es, que tengan menor evaluacion en la funcion objetivo
    que el admisible_treshold.
    
    Tambien devuelve los movimientos para encontrar cada uno
    
    '''
    
    #  obtener todo el vecindario completo, saber el movimiento de cada uno
    vecinos_movimiento = N2(S)
    
    # filtrar solo los admisibles
    # todos los vecinos tal que su movimiento no este en la lista tabu
    # o que sean mejores que el admisible_treshold
    adm_n_m = [v_m for v_m in vecinos_movimiento
               if v_m[1] not in tabu_list 
               or f(G, S=v_m[0]) < admisible_treshold]
    
    # devolve los vecinos admisibles con su movimiento
    return adm_n_m



# enocntrar el mejor vecino admisible
def best_admisible_N2(S, tabu_list, admisible_treshold, G):
    
    '''
    Toma una solucion, la lista tabu, y el criterio de aspiracion
    que es un treshold de la mejor solucion encontrada
    
    
    Encuentra el mejor vecino admisible de la soluicon actual
    '''
    
    # tomar todos los vecinos admisibles con sus movimientos
    vecinos_adm_mov = admisible_N2(S, 
                                   tabu_list = tabu_list,
                                   admisible_treshold = admisible_treshold,
                                   G = G)
    
    
    # ahorita que no se conoce cual es el mejor
    # decir que es el primero
    best_n_mov = vecinos_adm_mov[0]
    f_best_n = f(G, S=best_n_mov[0])
    
    
    # iterar en todos los vecinos admisibles
    for vecino_m in vecinos_adm_mov:
        
        # calcular que tan bueno es
        f_vecino = f(G, S=vecino_m[0])
        
        # si es el mejor hasta ahora
        if f_vecino < f_best_n:
            # entonce este es el mejor hasta ahora
            # actualizar
            best_n_mov = vecino_m
            f_best_n = f_vecino
            
            
    # se termina de ver los vecinos
    # se devuelve lo mejor encontrado
    return best_n_mov


# ------------------------------------------------------------------------


# actualizar la lista tabu

def actualizar_lista_tabu(lista_tabu, movimiento, k):
    
    '''
    Se pasa la lista tabu y el ultimo movimiento
    Se actualiza la lista tabu, es decir, 
    se mete el movimiento, y posiblemente
    se quita el mas viejo
    '''
    
    
    # poner el nuevo movimiento
    lista_tabu.append(movimiento)
    
    
    # si ya es mas grande que el tamaño permitido
    if len(lista_tabu) > k:
        
        # quitar el elementos mas viejo
        lista_tabu.pop(0)
        
    # devolver
    return lista_tabu
        



# ------------------------------------------------------------------------


# Funciones para graficar estadisticas del algoritmo

def graficar_f_S_and_f_best(hist, iteracion_inicial = 0):
    
    
    # sacar todas las metricas guardadas
    iteraciones = [stats[0] for stats in hist]
    f_S_evolucion = [stats[1] for stats in hist]
    f_best_evolucion = [stats[2] for stats in hist]
    
    
    # graficar a partir de una iteracion
    iteraciones = iteraciones[iteracion_inicial:]
    f_S_evolucion = f_S_evolucion[iteracion_inicial:]
    f_best_evolucion = f_best_evolucion[iteracion_inicial:]
    
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    
    # graficar la linea de la mejor solucion encontrada
    ax.plot(iteraciones, f_best_evolucion,
            label = "Mejor solucion encontrada",
            color = "red", linestyle = "--",
            alpha = 0.5)
    
    
    # graficar la linea de el costo de soluciones selecionada
    ax.plot(iteraciones, f_S_evolucion, 
            label = "Evaluacion mejor vecino admisible",
            alpha = 0.7)
    
    
        
    # indicar cosas
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Valor de la funcion objetivo ")
    
    
    # el titulo depende de si hay iteracion inicial
    if iteracion_inicial == 0:
        ax.set_title("Historial algoritmo")
    else:
        ax.set_title(f"Historial algoritmo desde la iteracion {iteracion_inicial}" )
        



    # Establecer el locator para el eje x para mostrar solo valores enteros
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.legend()
    
    plt.show()
    plt.close()


# ------------------------------------------------------------------------


### Tabu Search algorithm ###

### Simulated Annealing algorithm ###
def Tabu_Search_TSP(G, best_adm_Neighbor, k, max_iter, 
                    mostrar_graficar=True, iteracion_min_graficar=0):
    
    
    """
    Implementa el algoritmo de Tabu Search para TSP
    
    Parameters
    ----------
    G : NetworkX graph
        The graph representing the TSP instance.
    best_adm_Neighbor : function
        Return the best admissible neighbor of a given solution S,
        and the movement used to get it.
        It takes the arguments: (S, tabu_list, admisible_treshold, G)
        The aspiration criteria in selecting a tabu move is if it
        generates a solution better thatn the admisible_treshold
    k: integer
        Size of the tabu list
    max_iter: integer
        Stopping criteria. Number of iterations
    mostrar_graficar:  boolean
        Show graphs or not
    iteracion_min_graficar: integer
        Since which iteration to show the evolution
        
    
    El criterio de terminado del algoritmo es el numero maximo
    de iteraciones, despues de estas se termina.
    Se usa una lista tabu de tamaño k donde se guardan los movimientos.
    El Aspiration criteria es el siguiente:
        Se selecciona un vecino alcanzado con un movimiento
        de la lista tabu si es que este vecino
        es el mejor vecino encontrado hasta ahora.
        Esto se mide con un treshold, que es la evaluacion
        en la funcion objetivo del mejor vecino encontrado hasta ahora
    """
    
    print("Tabu Search algorithm")
    
    
    # el pos viene de las coordenadas de los nodos
    pos = nx.get_node_attributes(G, 'pos')
    
    # generar solucion inicial aleatoria
    S = S0_random(G)
    # obtener su evaluacion en la funcion
    f_S = f(G, S)
    
    # el mejor f hasta ahora pues es el f de la solucion inicial
    S_best = S
    f_best = f_S
    
    # mosrtrar la solucion inicial
    print(f"\n\nSolucion inicial S0 con: f(S_0) = {round(f_S, 4)}")


    # mostrar la solucion inicial
    if mostrar_graficar:
        DrawSolution(G, S, pos, "Solucion inicial")
    
    
    # ir guardando el número de iteración 
    # y el costo de la solución seleccionada,
    # tambien el costo de la mejor solucion encontrada
    # en la iteracion 0 porner lo de la solucion inicial
    hist = [(0, f_S, f_best)]
    
    
    # inicializar la tabu list vacia
    lista_tabu = []
    
    
    # hacer todas las iteracioens que se piden
    for idx_iteracion in range(1, max_iter + 1):
        
        # indicar
        print(f"\n\nIteracion: {idx_iteracion}")
        
        
        # tomar el mejor vecino admisible, con su movimiento
        S_prima, movimiento = best_adm_Neighbor(S, tabu_list= lista_tabu, 
                                                admisible_treshold= f_best,
                                                G=G)
        
        # ver que tan bueno es
        f_S_prima = f(G, S=S_prima)
        
        
        # informar
        print(f"El mejor vecino admisible tiene: f(S_prima) = {round(f_S_prima, 4)}")
        
        
        # ver el mejor vecino admisible
        if mostrar_graficar:
            DrawSolution(G, S_prima, pos, f"Mejor vecino admisible\nIteracion {idx_iteracion}")
        
        
        # ver si este vecino es el mejor hasta ahora
        if f_S_prima < f_best:
            
            # actualizar el nuevo mejor hasta ahora
            f_best = f_S_prima
            S_best = S_prima
            
            # indicarlo
            print("Es la mejor solucion explorada hasta ahora")
            
            
        # ver si este vecino es peor que la solucion actual
        if f_S_prima > f_S:
            print("Este vecino es peor que la actual, se escapa de posible optimo local")
        
        
        
        # moverse a este mejor vecino admisible
        S = S_prima
        f_S = f_S_prima
        
        
        # guardar los valores de esta iteracion
        hist.append((idx_iteracion, f_S, f_best))
        
        
        # actualizar la lista tabu
        lista_tabu = actualizar_lista_tabu(lista_tabu, movimiento, k)        
        
        
    
    # se terminan las iteraciones
    print(f"\n\nEl algoritmo termina despues de {max_iter} iteraciones")
    print(f"La ultima solucion alcanzada tiene: f(S) = {round(f_S, 4)}")
    print(f"La mejor solucion alcanzada tiene: f(S*) = {round(f_best, 4)}")
    
    
    # graficar
    if mostrar_graficar:
        
        graficar_f_S_and_f_best(hist, iteracion_min_graficar)
        
        
        # ver la ultima y la mejor
        DrawSolution(G, S, pos, f"Ultima solucion considerada")
        DrawSolution(G, S_best, pos, f"Mejor solucion encontrada")
    
    
    # devolver 
    return hist, f_S, f_best



# ----------------------------------------------------------------------------------


def checar_varias_k_para_N1(G, k_probar, num_iter = 100):
    
    '''
    Para el mismo Grafo, checar la mejor solucion alcanzada
    por varios valores de k
    
    Devolver el mejor valor de k
    '''
    
    # ir guardando lo mejor alcanzado
    mejores_alcanzados = []
    
    # iterar en los k
    for k in k_probar:
        
        # hacer la busqueda, tomar lo mejor alcanzado
        _, _, f_best_k = Tabu_Search_TSP(G, best_adm_Neighbor= best_admisible_N1, 
                                         k=k, max_iter=num_iter, mostrar_graficar = False)
        
        # agregar
        mejores_alcanzados.append(f_best_k)
        
    # graficar
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    

    
    # graficar la linea de el costo de soluciones selecionada
    ax.plot(k_probar, mejores_alcanzados)
    
    ax.set_xticks(k_probar)
    
    
        
    # indicar cosas
    ax.set_title("Desempeño distintos tamaños")
    ax.set_xlabel("Tamaño de lista tabu")
    ax.set_ylabel("Calidad solucion alcanzada")


    # Establecer el locator para el eje x para mostrar solo valores enteros
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
    plt.show()
    plt.close()
    
    # ver el indice donde esta el mejor k
    index_best_k = mejores_alcanzados.index(min(mejores_alcanzados))
    
    # devolverlo
    return k_probar[index_best_k]
        



# ----------------------------------------------------------------------------------



if __name__ == "__main__":
    
    # crear el grafo
    G = CreateGraph(100, seed=10)
    
    # numero de iteracioes
    numero_iteraciones = 100
    
    
    
    '''
    Busqueda tabu con el vecindatio N2
    
    Es bueno, pero es igual a solo hacer busqueda local para 100 iteraciones y N2
    '''
    
    
    # Hacer uno para ver que funciona
    # El tamaño maximo posible de la lista tabu es 4950, hay esos movimientos
    hist_N2, f_S_N2, f_best_N2 = Tabu_Search_TSP(G, 
                                                 best_adm_Neighbor= best_admisible_N2,
                                                 k= 2500, max_iter= numero_iteraciones,
                                                 mostrar_graficar = True, iteracion_min_graficar=0)
    
    # con 1000 iteraciones, con k=3000 se alcanza: 10.6. Graficar desde la 150
    # con 1000 iteraciones, con k=2500 se alcanza: 10.6 tambien
    # con 2000 iteraciones, con k=500 se alcanza: 9.6
    
    
    '''
    Busqueda tabu con el vecindatio N1
    
    No es tan bueno, pero es mejor que busqeuda local para 100 iteraciones y N1
    '''
    
    
    # probar desempeño varios valores k
    k_opciones = np.arange(0, 91, 2)
    # ver cual es el mejor
    best_k = checar_varias_k_para_N1(G, k_probar=k_opciones, num_iter= numero_iteraciones)
    
    
    
    # hacer una busqueda local con este valor de k
    hist_N1, f_S_N1, f_best_N1 = Tabu_Search_TSP(G, 
                                                 best_adm_Neighbor= best_admisible_N1,
                                                 k= best_k, max_iter= numero_iteraciones,
                                                 mostrar_graficar = True, iteracion_min_graficar=25)
    
    print("\n\nMejor k para la busqueda tabu con N1: " + str(best_k))
