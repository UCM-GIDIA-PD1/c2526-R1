from sklearn.dummy import DummyClassifier 
from training_utils import entrenamiento

if __name__ == '__main__':
    to_predict = "Made for kids"
    
    columns = ["Titulo", "Tags"] 
    
    project = "Baseline_kids"
    name = "V_Baseline_K"
    
    
    preprocess_type = "Bag of words" 

    params = {"strategy": ["most_frequent"]}
    
    score = "Precision" 
    average = "weighted"
    n_fold = 5 
    filtrado = False

    max_features = 5000
    ngram = (1,2)
    svd = 150

    entrenamiento(project, name, DummyClassifier, to_predict, max_features, ngram, svd, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)