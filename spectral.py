# Jakub Zbrzezny, nr indeksu: 286689
# Nick do publikacji wyników: Heihachi46

# Algorytm spektralny i jego implementacja

# Ze wzgledu na indeksowanie od 0 przez Pythona, zakladam, ze skupienia 
# i punkty numeruje sie od 0 do k - 1.

# 1) Macierz najblizszych sasiadow

import numpy as np        


# Funkcja pomocnicza liczaca odleglosc wektorow.

def odleglosc_wektorow(x, y):
    assert len(x) == len(y), "Dlugosci wektorow x i y nie sa rowne!"
    return np.sqrt(sum([(a - b) ** 2 for a,b in zip(x, y)]))


def Mnn(X, M):
    if (type(X) != np.ndarray): raise Exception("X nie jest macierza!")
    # X.shape[0] jest liczba wierszy macierzy X.
    assert M % 1 == 0, "M nie jest calkowite!"
    assert M >= 1 and M <= X.shape[0], "M jest mniejsze od 1 lub M jest wieksze niz liczba wierszy macierzy X!" 
    n = X.shape[0] # Liczba wierszy macierzy X.
    wsz_wiersze_S = [] # Bede po kolei dorzucac nowe wiersze do konstrukcji macierzy S.
    for i in range(n):
        odleglosci = np.linspace(0,0,n)
        wiersz_S = np.linspace(0,0,M) # W tej iteracji petli, tworze i-ty wiersz macierzy S.
        for j in range(n):
            odleglosci[j] = odleglosc_wektorow(X[i, ], X[j, ])
        # Chce uporzadkowac rosnaco odleglosci.
        posort_odleglosci = odleglosci
        posort_odleglosci = sorted(posort_odleglosci) # Funkcja sorted domyslnie sortuje rosnaco.
        for j in range(M): 
            for k in range(n):
                # Sprawdzam, czy to jest indeks j-tego najblizszego sasiada x_i wzgledem metryki euklidesowej.
                if(odleglosci[k] == posort_odleglosci[j]): 
                    wiersz_S[j] = k
        wsz_wiersze_S.append(wiersz_S) # Dorzucam nowy wiersz do macierzy S.
    S = np.array(wsz_wiersze_S)    
    return S



# 2) Macierz sasiedztwa

def Mnn_graph(S):
    if (type(S) != np.ndarray): raise Exception("S nie jest macierza!")
    n = S.shape[0]
    wsz_wiersze_Sas = []
    for i in range(n):
        wiersz_Sas = np.linspace(0, 0, n)
        for j in range(n):
            if any(S[i, ] == j) or any (S[j, ] == i):
                wiersz_Sas[j] = 1
            else:
                wiersz_Sas[j] = 0
        wsz_wiersze_Sas.append(wiersz_Sas)
    Sasiedzi = np.array(wsz_wiersze_Sas)
    return Sasiedzi



# 3) Laplasjan i jego wektory wlasne

def Laplacian_eigen(G, k):
    n = G.shape[0] # G jest macierza kwadratowa, zatem wystarczy mi tylko liczba wierszy.
    assert k % 1 == 0, "k nie jest calkowite!"
    assert k > 1, "k musi byc wieksze od 1!"
    # Wyznaczam macierz D
    stopnie = []
    for i in range(n):
        suma = 0
        for j in range(n):
            if j == i: continue
            suma += G[i, j] 
        stopnie.append(suma) 
    D = np.diag(stopnie)
    L = D - G    
    # np.linalg.eig(L) zwraca wektor, ktorego elementy sa wartosciami wlasnymi L
    # oraz macierz, ktorej kazdy wiersz jest wektorem wlasnym macierzy L.
    eig_vals, eig_vecs = np.linalg.eig(L) 
    sort_eig_vals = eig_vals
    sort_eig_vals = sorted(sort_eig_vals)
    wiersze_E = []
    for i in range(1, k + 1):
        wekt_wl_do_E = []
        for j in range(n):
            if(eig_vals[j] == sort_eig_vals[i]):
                wekt_wl_do_E = eig_vecs[j]
                break
        wiersze_E.append(wekt_wl_do_E)    
    E = np.array(wiersze_E)    
    E = np.transpose(E) # Dokonalem transpozycji E, poniewaz np.array(wektor_1, ..., wektor_n)
    # tworzy macierz o wierszach wektor_1, ..., wektor_n, a w zadaniu kolumny maja byc wektorami
    # wlasnymi, a nie wiersze.
    return E



# 4) Funkcja spectral_clustering(X, k, M).

pip install python-igraph # Uzyje pakietu igraph do algorytmu DFS.
import igraph as ig


def uspojnij(G): # Funkcja pomocnicza uspojniajaca graf sasiedztwa w przypadku, gdy graf nie jest spojny.
    # Szukam wszystkich skladowych spojnych za pomoca algorytmu przeszukiwania w głąb (DFS).
    # Jesli jest tylko jedna skladowa spojna, to znaczy, ze graf jest spojny. 
    # Wtedy nie modyfikuje macierzy G.
    
    n = G.shape[0]
    
    graf = ig.Graph.Adjacency(G.tolist())
    
    
    # Sprawdzam, czy graf jest spojny.
    # graf.dfs(0)[0] zwraca wszystkie odwiedzone wierzcholki grafu "graf", rozpoczynajac od wierzcholka 0.
    if (len(graf.dfs(0)[0]) == n): 
        return G
    
    # Uspojniam graf G.
    
    start = 0
    zbior = set(range(n)) # Jest to zbior wierzcholkow, z ktorego 
    # jeszcze nie uzylem skladowych do uspojnienia grafu G.
    while not(len(graf.dfs(0)[0]) == n):
        wierz_skl = graf.dfs(start)[0]
        i = wierz_skl[0] # Moge polaczyc dowolny wierzcholek z jednej skladowej i dowolny z drugiej,
        # wiec wezme pierwszy wierzcholek z skladowej "wierz_skl".
        zbior = zbior.difference(wierz_skl)
        # Szukam wierzcholka spoza skladowej zmodyfikowanego grafu G i wczesniejszych
        # skladowych uzytych do uspojnienia grafu G.
        for j in range(n):
            if j in zbior:
                G[i, j] = 1 # Lacze wierzcholek i z wierzcholkiem spoza skladowej "wierz_skl".
                G[j, i] = 1 # Mam przypadek grafu nieskierowanego, zatem lacze wierzcholki w obie strony.
                start = j
                break 
        graf = ig.Graph.Adjacency(G.tolist())
    
    return G
    
from sklearn.cluster import KMeans    

def spectral_clustering(X, k, M):
    assert k >= 2, "k musi byc wieksze, badz rowne 2!!!"
    S = Mnn(X, M)
    G = Mnn_graph(S)
    
    # Uspojniam graf sasiedztwa, jesli graf nie jest spojny.
    
    
    G = uspojnij(G)
    
    
    E = Laplacian_eigen(G, k)
    
    
    
    k_podzial = KMeans(n_clusters = k).fit(E)
    
    return k_podzial