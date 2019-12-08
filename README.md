# Spark project MS Big Data Télécom : Kickstarter campaigns 2019-2020
### Kaelig Castor

## Build and Run :

## TP1 (WordCount in scala) :
./build_and_submit.sh WordCount

## TP2 (Data Cleaning and Pre-Processing in scala) :
./build_and_submit.sh Preprocessor

## TP3 (Machine Learning Pipeline, Grid Search, Cross-Validation)
./build_and_submit.sh Trainer


|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 1892|
|           0|        1.0| 2310|
|           1|        1.0| 1589|
|           0|        0.0| 4963|

Modèle simple
f1score du modele simple (avant grid search) sur les donnees = 0.615

|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 1093|
|           0|        1.0| 2941|
|           1|        1.0| 2388|
|           0|        0.0| 4332|

Modèle paramétrique (grid search)
f1score du modele paramétrique (avec grid search) sur les données = 0.637
