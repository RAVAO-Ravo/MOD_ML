#!/bin/python3
#-*- coding:utf-8 -*-

# Importer les modules

import pandas as pd # Pandas pour la manipulation de dataframes
import matplotlib.pyplot as plt # Matplotlib pour visualisation graphique
import seaborn as sns # seaborn pour visualisation graphique


def data_info(dataset: pd.DataFrame) -> None :

	"Permet d'obtenir des infos sur le dataset."

	# Infos sur les dimensions du dataset
	print("DIMENSIONS :\n")
	dims = dataset.shape
	print("\tLe dataset contient {} échantillons, et {} features.\n".format(dims[0], dims[1]))
	print("\tLes features sont : {}.\n".format(dataset.columns.tolist()))

	# Infos sur les classes
	print("CLASSES :\n")
	classes = sorted(dataset.iloc[:, 4].value_counts().index.tolist())
	print("\tIl y a {} classes dans le dataset.\n".format(len(classes)))
	print("\tLes classes sont : {}.\n".format(classes))

	# Infos de duplications de données
	print("DUPLICATION :\n")
	duplicated_count = dataset.duplicated().value_counts().tolist()

	if len(duplicated_count) == 2 :

		duplicated_count = duplicated_count[1]

	else : 
		
		duplicated_count = 0

	print("\tIl y a {} lignes dupliquées dans le dataset.\n".format(duplicated_count))
	rows_duplicated = dataset[dataset.duplicated() == True].index.tolist()
	print("\tLes lignes dupliquées sont : {}.\n".format(rows_duplicated))


def stats_data(dataset: pd.DataFrame) -> None :

	"Permet de faire des stats à partir d'un dataset."

	# Statistiques globale du dataset
	print("STATISTIQUES GLOBALES : ")
	display(dataset.iloc[:, 0:4].describe())

	# Statistiques par espèce
	print("\nSTATISTIQUES PAR ESPÈCE")
	for i in sorted(dataset["target"].value_counts().index.tolist()) :

		print("\n\tEspèce : {}".format(i))
		display(dataset[dataset["target"] == i].iloc[:, 0:4].describe())


def boxplot_data(dataset: pd.DataFrame, figsize: tuple=(20, 20)) -> None :

	"Permet de créer un boxplot à partir d'un dataset."

	# Récupèrer les classes
	classes = sorted(dataset["target"].value_counts().index.tolist())

	# Récupèrer le nom des variables
	features_names = [col_name for col_name in dataset.columns.tolist() if col_name != "target"]

	# Définir le compteur des subplots
	cpt = 1

	# Définir une figure
	plt.figure(figsize=figsize)

	# Pour chaque variables
	for col_name in features_names :

		# Sur un suplot
		plt.subplot(len(features_names), len(classes), cpt)
		
		# Afficher le boxplot de la distribution des valeurs de la variable, en fonction des espèces
		sns.boxplot(data=dataset, x="target", y=col_name)

		# Passer au subplot suivant
		cpt+=1
	
	# Titrer la figure
	plt.suptitle("Boxplots des features en fonctions des espèces", y=0.90)

	# Afficher la figure
	plt.show()


def features_distributions(dataset: pd.DataFrame, figsize: tuple=(20, 20)) -> None :

	"Permet d'illustrer la correlation des features."
	
	# Récupérer le nom des variables
	features_names =  [col_name for col_name in dataset.columns.tolist() if col_name != "target"]
	
	# Récupérer le nombre de variables
	n_features = len(features_names)

	# Définir le compteur des subplots
	cpt = 1

	# Définir une figure
	plt.figure(figsize=figsize)

	# Afficher chaque variable en fonction des autres
	for i in features_names :

		for j in features_names :
			
			# Sur un subplot
			plt.subplot(n_features, n_features, cpt)

			# Afficher la distribution des espèces dans le plan
			plt.scatter(dataset.loc[:, i], dataset.loc[:, j], c=dataset["target"])

			# Nommer les axes
			plt.ylabel(i)
			plt.xlabel(j)

			# Passer au suivant
			cpt+=1

	# Afficher la figure
	plt.show()


def clean_duplicated(dataset: pd.DataFrame) -> pd.DataFrame :

	"Vérifie s'il y a des lignes dupliquées dans le dataset, et les retire."
	
	# Retirer les lignes dupliquées du dataset
	dataset = dataset[dataset.duplicated() == False].reset_index(drop=True)

	# Afficher quelques infos
	display(dataset.head())
	print("dimensions : {}".format(dataset.shape))

	return dataset