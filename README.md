# spam

Goal: Train a model to detect spam based on the Spambase Data Set (https://archive.ics.uci.edu/ml/datasets/Spambase).

## Run

To run self-contained:
```
pip3 install -r requirements.txt
python3 run.py
```

Or load `spam.ipynb` in ipython/jupyter notebook. This includes a small view of the dataset to get a feel for its complexity.

## Method
This analysis has the following core steps:

1. read in and prepare dataset
2. possibly apply dimensionality reduction (transform data)
3. split data into training and testing sets
4. train model on training set
5. test model by comparing expected and predicted results from the testing set
6. compute accuracy and error
7. pick winner

In this case we do not only want to use one algorithm, instead we use a Naive Bayes classifier and a Support Vector Machine (SVM). We also compare two decomposition methods: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). We compare the results when using all features of the data, as well as when only using the best and top ten decomposed features. This will tell us if all the features present in the dataset are actually predictive of the output. In total we compare 10 combinations.

For each combination run 15 iterations of training and testing, using a different subsets of the dataset (cross-validation). This ensures we don't accidentally pick a very homogeneous subset that doesn't support the features of the whole dataset.

## Results
The best result for the combinations given is an **accuracy of 0.913** with an error of 0.006. This is achieved using an **LDA** to reduce the dimensionality with 10 components and an **SVM** to train the classification model. Using 1 component is only marginally less accurate, this also depends on the seed, so results may vary slightly between runs. Generally, the Naive Bayes algorithm performs worse than the SVM but is trained much faster.

## Next steps
With the current setup, it is easy to add further classification algorithms and decompositions. Similary more combinations of components could be added by setting up an automatic grid search over the parameters.

## Appendix

On my machine, the following results are obtained

```
# Using all components.
## svm:
### Accuracy of 0.8373507057546146
### Error 0.003962395907138027
### Confusion Matrix
[[ 0.51646761  0.08975751]
 [ 0.07289178  0.3208831 ]]


## bayes:
### Accuracy of 0.8243937748823743
### Error 0.007406228497168452
### Confusion Matrix
[[ 0.44618169  0.15888527]
 [ 0.01672096  0.37821209]]


# Using pca
# Using 1 components.
## svm:
### Accuracy of 0.6944625407166125
### Error 0.005924731296883787
### Confusion Matrix
[[ 0.4723127   0.13637351]
 [ 0.16916395  0.22214984]]


## bayes:
### Accuracy of 0.6542164314151285
### Error 0.002000689269519709
### Confusion Matrix
[[ 0.58487152  0.02272892]
 [ 0.32305465  0.06934491]]


# Using lda
# Using 1 components.
## svm:
### Accuracy of 0.9109663409337676
### Error 0.004394348618690526
### Confusion Matrix
[[ 0.56728194  0.03923272]
 [ 0.04980094  0.3436844 ]]


## bayes:
### Accuracy of 0.8984437205935577
### Error 0.005735476906523811
### Confusion Matrix
[[ 0.57690916  0.02909881]
 [ 0.07245747  0.32153456]]


# Using pca
# Using 10 components.
## svm:
### Accuracy of 0.8014477017734347
### Error 0.016226206152889725
### Confusion Matrix
[[ 0.50503076  0.10025335]
 [ 0.09829895  0.29641694]]


## bayes:
### Accuracy of 0.6728193992037642
### Error 0.004899393765058115
### Confusion Matrix
[[ 0.58733261  0.01838581]
 [ 0.30879479  0.08548679]]


# Using lda
# Using 10 components.
## svm:
### Accuracy of 0.9130655085052479
### Error 0.006490451023549525
### Confusion Matrix
[[ 0.56293883  0.04010134]
 [ 0.04683315  0.35012667]]


## bayes:
### Accuracy of 0.8942453854505973
### Error 0.005786405828846596
### Confusion Matrix
[[ 0.57032211  0.03279045]
 [ 0.07296417  0.32392327]]


##############################
Best performing combination
Algorithm: svm
Components: 10
Decomposition: lda
Accuracy: 0.913066
Error: 0.006490
```
