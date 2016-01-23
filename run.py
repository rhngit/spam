from helper import lda, pca, svm, bayes
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from numpy import mean, var, sum, diag, shape

def load_data():
	'''
		Returns data and target from dataset
	'''
	data = genfromtxt('data/spambase/spambase.data', delimiter=',')
	target = data[:,-1]
	data = data[:,:-1]
	return data, target

def evaluate(algo, dim_rec, components, iterations=15):
	'''
		Returns average accuracy, error and confusion matrix for 
		combination of classification algorithm (algo), dimensionality reduction
		method (dim_rec) for n components, run iterations times.
	'''
	X, y = load_data()
	if components > 0:
		X, y = dim_rec(X, y, components)
	res = []
	for i in range(iterations):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		
		classifier = algo(X_train, y_train)
		confusion = 1.0 * classifier.classify(X_test, y_test) / len(X_test)
		res += [confusion]
	mean_confusion = mean(res, axis=0)
	var_confusion = var(res, axis=0)

	return sum(diag(mean_confusion)), iterations * sum(diag(var_confusion)), mean_confusion


if __name__ == "__main__":
	print ("##############################")
	print ("Starting")
	components = [0,1,10] # 0 will be handled like 'all'
	dim_rec = [pca, lda]
	algo = [svm, bayes]
	best = [0.0, 0.0, 'none', 'none', 0] # store best result
	for c in components:
		for d in dim_rec:
			print ("###########")
			tmp = 'all'
			if(c>0):
				tmp = str(c)
				print ("# Using %s" % (d.__name__))

			print ("# Using %s components." % (tmp))
			for a in algo:
				print ("## %s:" % a.__name__)
				acc, err, mat = evaluate(a, d, c)
				
				print ("### Accuracy of {n}".format(n=acc))
				print ("### Error {n}".format(n=err))
				print ("### Confusion Matrix")
				print (mat)
				print ()
				if acc > best[0]:
					best = [acc, err, d.__name__, a.__name__, c]
				print ()
			if(c==0):
				break

	print ("##############################")			
	print ("Best performing combination")
	print ("  Algorithm: %s" % (best[3]))
	print ("  Components: %s" % (best[4])) 
	print ("  Decomposition: %s" % (best[2])) 
	print ("  Accuracy: %f" % (best[0]))   
	print ("  Error: %f" % (best[1]))	  