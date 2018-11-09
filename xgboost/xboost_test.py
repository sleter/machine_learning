from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = loadtxt('test.csv', delimiter=',')
#one hot encoding for categorical data !!!

X = dataset[:,0:8]
Y = dataset[:,8]

#seed=?7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)#random_state=seed

# max_depth (int) – Maximum tree depth for base learners.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)
# n_estimators (int) – Number of boosted trees to fit.
# silent (boolean) – Whether to print messages while running boosting.
model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True)
model.fit(X_train, y_train)

#model.save_model('model01')

print(model)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



