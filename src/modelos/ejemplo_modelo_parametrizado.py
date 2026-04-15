from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier 
from xgboost import XGBClassifier
from comun.training_utils import entrenamiento
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo","Descripcion","Tags","Made for kids","Duracion","Titulo_canal"]


    max_features = 5000
    ngram = (1,2)
    svd = 150

    project = "Modelos definitivos generos"
    name = "Random Forest Generos 2"
    preprocess_type = "Word2Vec"

    # param_name = "n_neighbours"
    # param_vals = range(3,5)
    # metric = "cosine"

    params = {'criterion': ['gini'], 'max_depth': [30], 'max_features': ['sqrt'], 'n_estimators': [100]}

    score = "F1"
    average = "weighted"
    n_fold = 2

    # Opciones posibles de filtrado: 0 para no filtrar, 
    # 1 para filtrar videos con longitud extrema o sin información textual 
    # 2 para filtrar videos con subtítulos a None, dejando un poco de estos videos como ruido
    filtrado = 2
    
    entrenamiento(project, name, RandomForestClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                params, score, average, n_fold, filtrado)
