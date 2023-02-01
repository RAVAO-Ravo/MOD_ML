#!/bin/python3
#-*- coding:utf-8 -*-

# Importation des modules

import pandas as pd # Pandas pour la manipulation de dataframe
import numpy as np # Numpy pour les fonctions mathématiques
import matplotlib.pyplot as plt # Matplotlib pour les illustrations graphiques

# Fonctions pour évaluer des modèles
from sklearn.metrics import ConfusionMatrixDisplay, classification_report 
from sklearn.model_selection import learning_curve


def plot_LearningCurve(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, figsize: tuple=(10, 8)) -> None :

	"Affiche la courbe d'apprentissage d'un modèle."

	# Récupérer les éléments nécéssaires pour la création de la courbe d'apprentissage
	train_sizes, train_scores, val_scores = learning_curve(model, x_train, y_train, cv=3, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))

	# Récupérer les éléments pour la création de la "marge d'erreur"
	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)
	val_mean = np.mean(val_scores, axis=1)
	val_std = np.std(val_scores, axis=1)

	# Créer une figure
	plt.figure(figsize=figsize)

	# Afficher la courbe pour le train_set
	plt.plot(train_sizes, train_mean, '--', color="red", marker='+',  label="Training score")
	
	# Afficher la courbe pour le validation_set
	plt.plot(train_sizes, val_mean, color="blue", marker='o', label="Validation score")

	# Afficher la marge d'erreur du train_set
	plt.fill_between(train_sizes, (train_mean - train_std), (train_mean + train_std), color="yellow")
	
	# Afficher la marge d'erreur du test_set
	plt.fill_between(train_sizes, (val_mean - val_std), (val_mean + val_std), color="green")

	# Labeliser l'abscisse
	plt.xlabel("Training Size")
	
	# Labeliser l'ordonnées
	plt.ylabel("Accuracy") 

	# Titrer la figure
	plt.title("Learning Curve")

	# Légender la figure
	plt.legend(loc="lower right")

	# Ajouter une grille à la figure
	plt.grid()

	# Afficher la figure
	plt.show()


def plot_ConfMatrix(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, figsize: tuple=(12, 12)) -> None :
	
	"Permet le plotting de matrices de confusion, la première pour le TrainSet, et le second pour le TestSet."

	# Créer un figure
	plt.figure(figsize=figsize)

	# Afficher une matrice de confusion sur les données d'entraînement
	ConfusionMatrixDisplay.from_estimator(model, x_train, y_train, cmap=plt.cm.Blues, ax=plt.subplot(2, 2, 1))
	plt.title("Confusion matrix TrainningSet")
	
	# Afficher une matrice de confusion sur les données de test
	ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap=plt.cm.Blues, ax=plt.subplot(2, 2, 2))
	plt.title("Confusion matrix TestSet")

	# Afficher la figure
	plt.show()


def print_ClassReportRes(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None :

	"Affiche les résultats du classification report."

	# Afficher le rapport de classification sur les données d'entraînement
	print("\nTRAINING SET :\n")
	print(classification_report(model.predict(x_train), y_train))

	# Afficher le rapport de classification sur les données de test
	print("\nTEST SET :\n")
	print(classification_report(model.predict(x_test), y_test))
	print('\n')


def plot_ResTrainning(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, figsize: tuple=(12, 12)) -> None :

	"Affiche les résultats d'entraînement d'un modèle."

	# Transformer les labels pour éviter les message d'erreurs
	y_train = y_train.values.ravel()
	y_test = y_test.values.ravel()
	
	# Afficher le nom du modèle
	print(type(model).__name__ + " :")

	# Courbe d'apprentissage
	plot_LearningCurve(model, x_train, y_train)

	# Matrice de confusion
	plot_ConfMatrix(model, x_train, y_train, x_test, y_test, figsize)

	# Classification report
	print_ClassReportRes(model, x_train, y_train, x_test, y_test)