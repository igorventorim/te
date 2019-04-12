import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

# An advantage of naive Bayes is that it only requires a small number of 
# training data to estimate the parameters necessary for classification.
class BayesClassifier():

	def __init__(self,X,y,classname):
		self.X = X
		self.y = y
		self.classname = classname

	def run(self):
		print('Classifier: Bayes\n')
		skf = StratifiedKFold(n_splits=10, shuffle=True)

		y_pred_overall = []
		y_test_overall = []

		for train_index, test_index in skf.split(self.X, self.y):

			X_train, X_test = self.X[train_index], self.X[test_index]
			y_train, y_test = self.y[train_index], self.y[test_index]
			self.fit(X_train,y_train)
			y_pred = self.predict(X_test)

			y_pred_overall = np.concatenate([y_pred_overall, y_pred])
			y_test_overall = np.concatenate([y_test_overall, y_test])

		print('Bayes Classification Report: ')
		print (classification_report(y_test_overall, y_pred_overall, target_names=self.classname, digits=3))
		print('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
		print('Bayes Confusion Matrix: ')
		print (confusion_matrix(y_test_overall, y_pred_overall))

	def fit(self,X_train,y_train):
		classes = {}
		for x in zip(X_train,y_train):
			if x[1] in classes:
				classes[x[1]].append(x[0])
			else:
				classes[x[1]] = [x[0]]
		averages = {}
		for key in classes:
			average = np.average(classes[key],axis=0)
			variance = np.var(classes[key],axis=0)

		import ipdb; ipdb.set_trace()

	def predict(self,X_test):
		pass


