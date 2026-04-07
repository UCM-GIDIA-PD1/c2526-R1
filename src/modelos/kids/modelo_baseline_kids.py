from sklearn.dummy import DummyClassifier 
from comun.training_utils import entrenamiento

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

    entrenamiento(project, name, DummyClassifier, to_predict, preprocess_type, columns, 
                  params, score, average, n_fold, filtrado)