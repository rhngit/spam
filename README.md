# spam

Goal: Train a model to detect spam based on the Spambase Data Set (https://archive.ics.uci.edu/ml/datasets/Spambase).

## Run

To run self-contained:
```
pip install -r requirements.txt
python3 run.py
```

Or load `spam.ipynb` in ipython/jupyter notebook.

## Results

The best result for the combinations given is an accuracy of 0.915 with an error of 0.007. This is achieved using an LDA to reduce the dimensionality with 1 component and an SVM to train the classification model. Using 10 components is only marginally less accurate.



On my machine, the following results are obtained

```
##############################
###########
# Using all components.
## svm:
### Accuracy of 0.8338038364096996
### Error 0.0038992063183301254
### Confusion Matrix
[[ 0.51082157  0.09156714]
 [ 0.07462903  0.32298227]]


## bayes:
### Accuracy of 0.8191820484980095
### Error 0.004448892865746091
### Confusion Matrix
[[ 0.44060803  0.1655447 ]
 [ 0.01527325  0.37857401]]


###########
# Using pca
# Using 1 components.
## svm:
### Accuracy of 0.6951863916033298
### Error 0.00450359430083353
### Confusion Matrix
[[ 0.47209555  0.13630112]
 [ 0.16851249  0.22309084]]


## bayes:
### Accuracy of 0.6581252262034021
### Error 0.002980913836201427
### Confusion Matrix
[[ 0.58168657  0.02468332]
 [ 0.31719146  0.07643865]]


###########
# Using lda
# Using 1 components.
## svm:
### Accuracy of 0.9148751357220413
### Error 0.006605041098775785
### Confusion Matrix
[[ 0.56518277  0.03756786]
 [ 0.047557    0.34969236]]


## bayes:
### Accuracy of 0.8959826275787188
### Error 0.004036588658176151
### Confusion Matrix
[[ 0.57806732  0.02823018]
 [ 0.07578719  0.31791531]]


###########
# Using pca
# Using 10 components.
## svm:
### Accuracy of 0.7957292797683677
### Error 0.009791242504586944
### Confusion Matrix
[[ 0.49337676  0.11653999]
 [ 0.08773073  0.30235252]]


## bayes:
### Accuracy of 0.6741223307998553
### Error 0.006834535625292031
### Confusion Matrix
[[ 0.58284473  0.01910966]
 [ 0.30676801  0.0912776 ]]


###########
# Using lda
# Using 10 components.
## svm:
### Accuracy of 0.9116901918204849
### Error 0.00554386469568594
### Confusion Matrix
[[ 0.56525516  0.04024611]
 [ 0.0480637   0.34643503]]


## bayes:
### Accuracy of 0.8982265653275425
### Error 0.0039669543600619785
### Confusion Matrix
[[ 0.57835686  0.02844734]
 [ 0.07332609  0.31986971]]

##############################
Best performing combination
  Algorithm: svm
  Components: 1
  Decomposition: lda
  Accuracy: 0.914875
  Error: 0.006605
```
