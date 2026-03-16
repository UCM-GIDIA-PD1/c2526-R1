from sklearn.neighbors import KNeighborsClassifier
from modelos_utils import entrenamiento

if __name__ == '__main__':
    entrenamiento(KNeighborsClassifier, "Generos", "Word2Vec", "n_neighbors", range(3,4), "cosine", 2)
