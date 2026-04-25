'''
Entrenar un modelo sencillo para posteriormente desplegarlo como API REST.
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

# semilla
RND_SEED = 42

# datos
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND_SEED, shuffle=True)

# entrenar clasificador
clf = DecisionTreeClassifier(random_state=RND_SEED)
clf.fit(X_train, y_train)

# medir el error
print('train accuracy:', clf.score(X_train, y_train))
print('test accuracy:', clf.score(X_test, y_test)) 

# hacer una predicción
pred_y = clf.predict([X_test[0]])[0]
probs = clf.predict_proba([X_test[0]])[0]
print()
print('sample:', X_test[0])
print('real label:', y_test[0])
print('pred label:', pred_y)
print('probabilities:', probs)

# guardar el modelo
dump(clf, 'models/final_model.joblib')