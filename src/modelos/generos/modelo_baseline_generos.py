from sklearn.dummy import DummyClassifier 
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Generos"
    
    columns = ["Titulo", "Tags"] 
    
    project = "Baseline_generos"
    name = "V_Baseline_G"
    
    
    preprocess_type = "Bag of words" 


    params = {"strategy": ["most_frequent"]}
    
    score = "F1" 
    average = "weighted"
    n_fold = 5 
    filtrado = False

    max_features = 5000
    ngram = (1,2)
    svd = 150

    entrenamiento(project, name, DummyClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)