import pandas as pd
import numpy as np
df = pd.read_json('data.json')
X = df['imgData']
X = pd.DataFrame(X.tolist(), index= X.index)
y = df['class']
# for a in range(len(X)):
#     X[a] = np.asmatrix(X[a]).reshape((64,64))
# print(X[:0])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LogisticRegression().fit(X_train,y_train)
# model.predict()
y_pre = model.predict(X_test)
print(classification_report(y_pre, y_test, digits=3))
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))