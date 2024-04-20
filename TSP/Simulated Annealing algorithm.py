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

def CreateGraph(n=10):
    """
    Create a complete graph with n nodes randomly located in a 2D space.
    Each edge is assigned its length between its endpoints.
    """
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
            node_size=100, node_color='darkcyan')
    
    
    # calcular el costo
    cost = np.round(f(G, S), 2)
    
    # titulo de la figura
    ax.set_title(text + f'\nCost: {cost}')
    
    plt.show()
    plt.close()



# dibuja la exploracion de las soluciones
def DrawGraphSolution(G_solutions, moves):
    """
    Draw the solution graph, where each node is a solution and
    the edges represent movement from one solution to another.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.spring_layout(G_solutions, iterations=100, k=0.3)
    node_colors = ['red' if list(v) in moves else 'white' for v in G_solutions.nodes]
    ax.set_facecolor('black')
    nx.draw(G_solutions, pos= pos, ax=ax,
            edgecolors='black', node_color=node_colors, node_size=1000)
    node_labels = {v: f'${round(f, 2)}$' for v, f in nx.get_node_attributes(G_solutions, 'f').items()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, 
                            font_size=14, font_color='black')
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


# para las funciones N_i
# si first esta activado entonces devuelven un vecino aleatorio
# si first es falso entonces devuelve todos los vecinos

def move_n1(S, i):
    """
    Swap the positions of two consecutive indices of a tour.
    """
    S_prime = S.copy()
    S_prime[i-1], S_prime[i] = S[i], S[i-1]
    return S_prime


def N1(S, first):
    """
    Generate neighbors of a solution by applying
    move N1 to each pair of consecutive indices.
    """
    neighbor = []
    n = len(S)
    if first is True:
        i = random.randint(1, n-1)
        S_prime = move_n1(S, i)
        neighbor.append(S_prime)
    else:
        for i in range(1, n):
            S_prime = move_n1(S, i)
            neighbor.append(S_prime)
    return neighbor


def move_n2(S, i, j):
    """
    Swap the positions of any two indices in a tour
    """
    S_prime = S.copy()
    S_prime[i], S_prime[j] = S[j], S[i]
    return S_prime


def N2(S, first):
    """
    Generate all neighbors of a solution by applying
    move N2 to each pair of distinct indices.
    """
    n = len(S)
    neighbor = []
    combinations_ = list(combinations(range(n), 2))
    if first is True:
        i, j = random.sample(combinations_, 1)[0]
        S_prime = move_n2(S, i, j)
        neighbor.append(S_prime)
    else:
        for i, j in combinations_:
            S_prime = move_n2(S, i, j)
            neighbor.append(S_prime)
    return neighbor


def move_n3(S, i, j):
    """
    Move the element at index i to index j in the tour, and
    rearranging intermediate elements.
    """
    S_prime = S.copy()
    v = S_prime.pop(i)
    S_prime.insert(j, v)
    return S_prime

    
def N3(S, first):
    """
    Generates all neighbors of a solution by applying
    move N3 to each pair of distinct indices.
    """
    n = len(S)
    neighbor = []
    combinations_ = list(combinations(range(n), 2))
    if first is True:
        i, j = random.sample(combinations_, 1)[0]
        S_prime = move_n3(S, i, j)
        neighbor.append(S_prime)
    else:
        for i, j in combinations_:
            S_prime = move_n3(S, i, j)
            neighbor.append(S_prime) 
    return neighbor


def move_2_opt(S, i, j):
    """
    Swap elements at indices i+1 and j.
    """
    S_prime = S.copy()
    S_prime[i+1] = S[j]
    S_prime[j] = S[i+1]
    return S_prime


def N_2_opt(S, first):
    """
    Generate all neighbors of a solution by applying
    move 2-Opt to each pair of  non-adjacent indices.
    """
    n = len(S)
    neighbor = []
    combinations_ = list(combinations(range(n), 2))
    combinations_ = [(i, j) for i, j in combinations_ if (i<j-1) | (i>j+1)]
    if first is True:
        i, j = random.sample(combinations_, 1)[0]
        S_prime = move_2_opt(S, i, j)
        neighbor.append(S_prime)
    else:
        for i, j in combinations_:
            S_prime = move_2_opt(S, i, j)
            neighbor.append(S_prime) 
    return neighbor


# ------------------------------------------------------------------------

# Cooling schedule options


def g_linear(t, beta = 0.001):
    
    assert beta > 0
    
    return t - beta



def g_geometric(t, alpha = 0.75):
    
    assert alpha > 0
    assert alpha < 1
    
    return t*alpha


# ------------------------------------------------------------------------


### Simulated Annealing algorithm ###
def Simulated_Anealing_TSP(G, move, T_max, T_min, g_temp, n):
    
    
    """
    Implementa el algoritmo de simulated annealing para TSP
    
    Parameters
    ----------
    G : NetworkX graph
        The graph representing the TSP instance.
    move : function
        The neighborhood move function to be used in the local search. 
    T_max: float
        Initial temperature
    T_min: float
        Stopping criteria
    g_temp: function
        cooling schedule (see the options above)
    n: integer
        Number of iterations for each temperature
        
    
    Se inicia con la temperatura T_max,
    por cada temperatura, se hacen n iteraciones.
    Despues se decrese la temperatura con: T_new = g_temp(T_actual),
    y se hacen n iteraciones, se repite este proceso.
    Al algoritmo se detiene hasta que se sobrepasa la
    temperatura minima, este es el Stopping criteria: T_actual < T_min
    
    """
    
    print("Simulated Annealing algorithm")
    
    
    # el pos viene de las coordenadas de los nodos
    pos = nx.get_node_attributes(G, 'pos')
    
    # generar solucion inicial aleatoria
    S = S0_random(G)
    # obtener su evaluacion en la funcion
    f_S = f(G, S)
        
    # mosrtrar esto
    print(f"\n\nSolucion inicial S0 con: f(S_0) = {round(f_S, 4)}")
    
    # mostrar la solucion inicial
    DrawSolution(G, S, pos, "Solucion inicial")
        
    
    # comenzar con la temperatura inicial
    T_actual = T_max
    
    # ir contando el numero de iteracion
    num_interacion = 1
    
    # ir guardando el numero de iteracion, temperatura, delta E y f(S)
    # en una lista
    hist = [(0, T_max, 0, f_S)]
    
    # crear el grafo de soluciones vacio
    # son las soluciones que se van explorando
    G_solutions = nx.Graph()
    
    # añadir la solucion iniical al grafo
    G_solutions.add_node(tuple(S), f= f_S)
    
    # guardar la solucion inicial en las soluciones que se seleccionan
    move_solutions = [S]
    
    # variable para ver cuando se continua el algoritmo
    continuar = True
    
    # repetir mientras se quiera
    while(continuar):
        
        
        # para esta temperatura hacer n iteraciones
        for i in range(n):
            
            # indicar en que iteracion esta, con que temperatura
            print(f"\n\nIteracion: {num_interacion}")
            print(f"Temperatura actual: {round(T_actual, 4)}")
            
            
            # tomar un vecino aleatorio
            # poner first=True para que sea uno aleatorio
            S_prima = move(S, first=True)[0]
            
            # ver su evaluacion en la funcion
            f_S_prima = f(G, S_prima)
            
            
            # añadir esta solucion a todas las soluciones exploradas
            G_solutions.add_node(tuple(S_prima), f= f_S_prima)
            # añadir el eje de desde donde se exploto
            G_solutions.add_edge(tuple(S), tuple(S_prima))
            
            # indicar esto
            print("Se encuentra un vecino aleatorio")
            print(f"Para este vecino f(S_prima) = {round(f_S_prima, 4)}")
            
            # calcular el delta_E
            delta_E = f_S_prima - f_S
            
            # imprimir
            print(f"Se tiene que ΔE = {round(delta_E, 4)}")
            
            
            # si el vecino es mejor que el actual
            if delta_E < 0:
                
                # decir que es mejor
                print("Este vecino es mejor que la solucion actual")
                print("Se acepta el vecino")
                
                # pasarse a ese vecino
                S = S_prima
                f_S = f_S_prima
                
                # verlo
                DrawSolution(G, S, pos, f"Vecino aceptado\nIteracion {num_interacion}")
                
                
                # guardarla en las soluciones que se seleccionan
                move_solutions.append(S)
                
                
            # el vecino es peor que la actual
            else:
                
                print("Este vecino es peor que la solucion actual")
                
                # ver la probablidad de aceptarlo
                proba_aceptar  = math.exp((-delta_E)/T_actual)
                
                print(f"La probabilidad de aceptarlo es: {round(proba_aceptar, 4)}")
                
                # ver si se acepta
                if np.random.uniform(0, 1) < proba_aceptar:
                    
                    print("Se acepta el vecino")
                    # pasarse a ese vecino
                    S = S_prima
                    f_S = f_S_prima
                    
                    # verlo
                    DrawSolution(G, S, pos, f"Vecino aceptado\nIteracion {num_interacion}")
                    
                    # guardarla en las soluciones que se seleccionan
                    move_solutions.append(S)
                    
                # no se acepta
                else:
                    
                    print("No se acepta, se sigue con la misma solucion")
                    
                    
                
                    
            # guardar la informacion de esta iteracion
            hist.append((num_interacion, T_actual, delta_E, f_S))
            
            
            # actualizar el numero de iteracion para la proxima iteracion
            num_interacion += 1
            
            # end for
        
        
        # actualizar la temperatura
        T_actual = g_temp(T_actual)
        
        # ver si si se va a hacer la proxima iteracion
        if T_actual < T_min:
            print(f"\n\nSiguiente temperatura: {round(T_actual, 4)}")
            print(f"Es menor a la menor temperatura aceptada: {round(T_min, 4)}")
            print(f"Se termina el algoritmo despues de {num_interacion-1} iteraciones")
            
            # terminar
            continuar = False
            
            
    # se termina el while
    
    # indicar la solucion encontrada
    print(f"\n\nSe encontro una solucion S con: f(S) = {round(f_S)}")
    
    # ver la solucion encontrada
    DrawSolution(G, S, pos, f"Solucion Encontrada")
    
    # mostrar la evolucion de la temperatura
    graficar_temperatura(hist)
    
    
    # mostrar los delta_E a traves de las iteraciones
    graficar_delta_E(hist)
    
    
    # mostrar la evaluacion de la funcion objetivo a travez del tiempo
    graficar_f_S(hist)
    
    # mostrar las soluciones expploradas, y las que se seleccionaron
    DrawGraphSolution(G_solutions, move_solutions)
    
    
    
    # devolver el optimo local
    return f_S, hist, num_interacion-1
    


# ------------------------------------------------------------------------


# Funciones para graficar estadisticas del algoritmo


def graficar_delta_E(hist):
    
    # sacar las delta_E
    deltas_E = [stats[2] for stats in hist]
    
    # sacar las iteraciones
    iteraciones = [stats[0] for stats in hist]
    
    # la de la solucion iniicla obviamente se quita
    deltas_E = deltas_E[1:]
    iteraciones = iteraciones[1:]
    
        
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    
    # graficar la linea
    ax.scatter(iteraciones, deltas_E)
    
    # hacer una linea horizontal en 0
    ax.axhline(y=0, color="gray", linestyle="--")
    
    
        
    # indicar cosas
    ax.set_title("Cambios de soluciones encontradas")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("ΔE")


    # Establecer el locator para el eje x para mostrar solo valores enteros
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.show()
    plt.close()



def graficar_temperatura(hist):
    
    # sacar las temperatudas
    temperaturas = [stats[1] for stats in hist]
    
    # sacar las iteraciones
    iteraciones = [stats[0] for stats in hist]
    
    # la de la solucion iniicla obviamente se quita
    temperaturas = temperaturas[1:]
    iteraciones = iteraciones[1:]
    
        
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    
    # graficar la linea
    ax.plot(iteraciones, temperaturas)
    
    
        
    # indicar cosas
    ax.set_title("Cambio de la temperatura en el algoritmo")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Temperatura")


    # Establecer el locator para el eje x para mostrar solo valores enteros
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.show()
    plt.close()



def graficar_f_S(hist):
    
    # sacar todas las evaluaciones de la funcion
    hist_f = [stats[3] for stats in hist]
    
    # sacar las iteraciones
    iteraciones = [stats[0] for stats in hist]
    
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    
    # graficar la linea
    ax.plot(iteraciones, hist_f)
    
    
        
    # indicar cosas
    ax.set_title("Historial de funcion objetivo durante el algoritmo")
    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Funcion objetivo en la solucion actual")


    # Establecer el locator para el eje x para mostrar solo valores enteros
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.show()
    plt.close()


# ------------------------------------------------------------------------

if __name__ == "__main__":

    
    # crear el grafo
    G = CreateGraph(10)
    
    
    # combinaciones temperatura a probar. (T_max, T_min)
    combinaciones_t = [(1, 0.00001), (0.5, 0.000001), (10, 2), (3, 0.0001), (0.5, 0.0001),
                       (1, 0.0005), (0.00001, 0.0000001), (0.7, 0.00002),
                       (4, 0.002), (4, 0.000001)]
    
    # ver que sean 10
    assert len(combinaciones_t) == 10
    
    # ir guardando las evaluaciones de la funcion y el numero de iteraciones
    evaluaciones_funcion = []
    numeros_it = []
    
    # iterar en las combinaciones de temperatura
    for temperaturas_probar in combinaciones_t:
        
        
        # hacer la busqueda local
        f_S, hist, num_iteraciones = Simulated_Anealing_TSP(G, move=N2, 
                                              T_max= temperaturas_probar[0],
                                              T_min= temperaturas_probar[1],
                                              g_temp=g_geometric, n = 10)
        
        # agregar las cosas de interes
        evaluaciones_funcion.append(f_S)
        numeros_it.append(num_iteraciones)
        
        
    # hacer el dataframe con los datos
    df = pd.DataFrame()
    
    # agregar los datos de interes
    df['T_max y T_min'] = combinaciones_t
    df['Numero de iteracioens'] = numeros_it
    df['Calidad de solucion'] = evaluaciones_funcion
    
    # guardar el df
    df.to_csv('tabla.csv', index=False)










