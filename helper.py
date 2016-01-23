# import algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# import decompositions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

# Start with Decompositions
def lda(X, y, n):
	'''
		Returns optimal projection of the data
		LDA with n components
	'''
	selector = LinearDiscriminantAnalysis(n_components=n)
	selector.fit(X, y)
	return selector.transform(X), y

def pca(X, y, n):
	'''
		Returns optimal projection of the data
		PCA with n components
	'''
	selector = PCA(n_components=n)
	selector.fit(X, y)
	return selector.transform(X), y

class svm():
	def __init__(self, X_train, y_train):
		self.classifier = SVC(kernel='rbf')
		self.classifier.fit(X_train, y_train)

	def classify(self, X_test, y_test):
		pred = self.classifier.predict(X_test)
		return confusion_matrix(y_test, pred)

class bayes():
	def __init__(self, X_train, y_train):
		self.classifier = GaussianNB()
		self.classifier.fit(X_train, y_train)

	def classify(self, X_test, y_test):
		pred = self.classifier.predict(X_test)
		return confusion_matrix(y_test, pred)