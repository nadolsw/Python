#First install scki kit learn by typing 'pip install -U scikit-learn' into windows command prompt (NOT PYTHON IDE)#

import sklearn
from sklearn import tree

DT = tree.DecisionTreeClassifier()

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], 
    [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
    [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

DT = DT.fit(X, Y)

prediction = DT.predict([[120, 50, 33]])

print(prediction)

print DT.predict_proba([[120, 50, 33]])
