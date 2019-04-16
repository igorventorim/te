import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

# An advantage of naive Bayes is that it only requires a small number of 
# training data to estimate the parameters necessary for classification.
class BayesClassifier():

	def __init__(self,X,y,classname,priors=None):
		#self.SMALL_SAMPLE_CORRECTION = 0.0000001
		#REMOVE LAST ELEMENT
		self.X = X.T[:-1].T
		self.y = y
		self.classname = classname
		
		#Train parameters
		self.averages = {}
		self.variances = {}
		self.covariance_matrix = {}
		self.d = 0
		if priors is None:
			self.priors = {}
		else:
			self.priors = priors

	def run(self):
		print('Classifier: Gaussian Naive Bayes\n')
		skf = StratifiedKFold(n_splits=10, shuffle=True)
		y_pred_overall = []
		y_test_overall = []

		for train_index, test_index in skf.split(self.X, self.y):

			X_train, X_test = self.X[train_index], self.X[test_index]
			y_train, y_test = self.y[train_index], self.y[test_index]
			self.fit(X_train,y_train)
			y_pred = self.predict(X_test)
			# import ipdb; ipdb.set_trace()
			y_pred_overall = np.concatenate([y_pred_overall, y_pred])
			y_test_overall = np.concatenate([y_test_overall, y_test])

		print('Bayes Classification Report: ')
		print (classification_report(y_test_overall, y_pred_overall, target_names=self.classname, digits=3))
		print('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
		print('Bayes Confusion Matrix: ')
		print (confusion_matrix(y_test_overall, y_pred_overall))
		print("\n\n\n")


	def fit(self,X_train,y_train,priors=None):
		classes = {}

		#Classes split
		for x in zip(X_train,y_train):
			# REMOVE LAST ELEMENT [:-1]
			if x[1] in classes:
				classes[x[1]].append(x[0])
			else:
				classes[x[1]] = [x[0]]

		#REMOVE THE LAST FEATURE OF THE COUNT
		self.d = len(X_train[0])
		
		#Setting classifier parameters
		for key in classes:
			#Adding average of class
			self.averages[key] = np.average(classes[key],axis=0) #+ self.SMALL_SAMPLE_CORRECTION

			#Adding variance of class
			self.variances[key] = np.var(classes[key],axis=0) #+ self.SMALL_SAMPLE_CORRECTION
			
			self.covariance_matrix[key] = np.cov(np.array(classes[key]),rowvar=False)

			#Set equiprobable classes
			if priors is None:
				self.priors[key] = 1 / float(len(classes))
		

	def predict(self,X_test):
		y_pred = []
		for sample in X_test:
			posterior = []
			for label in range(0,len(self.classname)):
				# posterior.append(self.gaussian_post(label,sample))
				posterior.append(self.qda(label,sample))
			posterior = np.array(posterior)
			# import ipdb; ipdb.set_trace()
			# y_pred.append(self.classname[posterior.argmax()])
			y_pred.append(posterior.argmax())

		return y_pred


	def gaussian_post(self,classlabel,sample):
		term_a = (1/(np.sqrt(2*np.pi*np.array(self.variances[classlabel]))))
		term_b = np.exp(-((np.array(sample)-np.array(self.averages[classlabel]))**2)/(2*np.array(self.variances[classlabel])))
		return np.prod(term_a *term_b) * self.priors[classlabel]		

	#Quadratic Discriminant Analysis
	def qda(self,classlabel,sample):

		term_a = (1/(((2*np.pi)**(self.d/float(2)))*(np.linalg.det(self.covariance_matrix[classlabel])**(1/float(2)))))
		term_b = np.exp(-1/float(2) * ( np.dot(np.dot((sample - self.averages[classlabel]).T, np.linalg.inv(self.covariance_matrix[classlabel])), (sample - self.averages[classlabel]))))

		# term_1 = -(1/float(2)) * sample.T * np.linalg.inv(self.covariance_matrix[classlabel]) * sample
		# term_2 = sample.T * np.linalg.inv(self.covariance_matrix[classlabel]) * self.averages[classlabel]
		# term_3 = - 1/float(2)*self.averages[classlabel].T * np.linalg.inv(self.covariance_matrix[classlabel])*self.averages[classlabel]
		# term_4 = -(1/float(2))*np.log(np.linalg.det(self.covariance_matrix[classlabel])) 
		# term_5 =  np.log(np.pi*self.priors[classlabel])

		return term_a*term_b * self.priors[classlabel]
