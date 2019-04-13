import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class QDA_SKLEARN():

	def __init__(self,X,y,classname):
		self.X = X
		self.y = y
		self.classname = classname

	def run(self):
		clf = QuadraticDiscriminantAnalysis()


		print('Classifier: QDA SKLEARN \n')
		skf = StratifiedKFold(n_splits=10, shuffle=True)

		y_pred_overall = []
		y_test_overall = []

		for train_index, test_index in skf.split(self.X, self.y):

			X_train, X_test = self.X[train_index], self.X[test_index]
			y_train, y_test = self.y[train_index], self.y[test_index]
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)
			y_pred_overall = np.concatenate([y_pred_overall, y_pred])
			y_test_overall = np.concatenate([y_test_overall, y_test])

		print('QDA SKLEARN Classification Report: ')
		print (classification_report(y_test_overall, y_pred_overall, target_names=self.classname, digits=3))
		print('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
		print('QDA SKLEARN Confusion Matrix: ')
		print (confusion_matrix(y_test_overall, y_pred_overall))
		print("\n\n\n")
