#!/bin/python3
#-*- coding:utf-8 -*-

# Importer les modules

import pandas as pd # Pandas pour la manipulation de dataframe
import numpy as np # Numpy pour générer les données artificiels

from sklearn.datasets import load_iris # On travaille sur les données d'Iris
from sklearn.model_selection import train_test_split # Pour splitter les données
from twinning import twin # Pour splitter les données (technique de splitting déterministe)


def create_sample(dataset: pd.DataFrame, n_sample: int=1000) -> pd.DataFrame :

	"Permet de générer des données artificiels."

	# Récupérer la liste triée des espèces présentes 
	species = sorted(dataset["target"].value_counts().index.tolist())

	# Initialiser la variable (ou dataframe) résultat
	df_res = None

	# Pour chaque espèce
	for specie in species :

		# Récupérer les statistiques correspondantes, sous forme de dataframe
		tmp = dataset[dataset["target"] == specie].iloc[:, 0:4].describe()

		# Initialiser une variable temporaire
		df_tmp = None

		# Pour chaque colonne du dataframe statistique
		for col_name in tmp.columns.tolist() :

			# Récupérer la moyenne, l'écart-type, le max, et le min de la colonne
			moy = tmp.at["mean", (col_name)]
			ec_type = tmp.at["std", (col_name)]
			percent_max = tmp.at["max", (col_name)]
			percent_min = tmp.at["min", (col_name)]

			# Initialiser la liste qui contiendra les valeurs générées
			data = []

			# Initialiser un compteur
			cpt = 0

			# Tant que l'on n'a pas généré tout les données
			while cpt != n_sample :
				
				# Générer une donnée, selon la loi normale de paramètre moyenne, écart-type
				value = np.random.normal(moy, ec_type, 1).tolist()[0]

				# Vérifier que la valeur se situe entre le minimum et le maximum
				if percent_min < value and value < percent_max :

					# L'ajouter à la liste
					data.append(value)

					# Incrémenter le compteur
					cpt+=1

			# S'il sagit de la première colonne
			if type(df_tmp) != pd.DataFrame :

				# Transformer la variable temporaire en dataframe temporaire
				df_tmp = pd.DataFrame(data={col_name : [abs(value) for value in data]})

			# Si ce n'est pas la première colonne
			else : 
				
				# Concaténer le dataframe temporaire avec les nouvelles données, sur l'axe 1
				df_tmp = pd.concat([df_tmp, pd.DataFrame(data={col_name : [abs(value) for value in data]})], axis=1)
		
		# Ajouter la colonne "target" au dataframe temporaire
		df_tmp = pd.concat([df_tmp, pd.DataFrame(data={"target" : [specie for _ in range(0, n_sample, 1)]})], axis=1)

		# S'il sagit de la première espèce
		if type(df_res) != pd.DataFrame :
			
			# Affecter à la variable résultat le dataframe temporaire
			df_res = df_tmp
		
		# Si ce n'est pas la première espèce
		else :
			
			# Concaténer, sur l'axe 0, le dataframe temporaire au dataframe résultat
			df_res = pd.concat([df_res, df_tmp], axis=0)

	# Retourner le dataframe résultat
	return df_res

if __name__ == "__main__" :

	# Charger les données d'Iris, sous forme de tuple
	features, labels = load_iris(return_X_y=True, as_frame=True)

	# Concaténer les features et les labels
	dataset = pd.concat([features, labels], axis=1)
	
	# Définir l'inverse du ratio de splitting
	r = round(1 / (1/3))
	
	# Récupérer les index pour le test_set, sous forme de liste
	index_test = twin(dataset.to_numpy(), r, u1=42).tolist()

	# Récupérer les index pour le train_set, sous forme de liste
	index_train = [i for i in range(0, len(dataset), 1) if i not in index_test]

	# Splitter les données d'Iris, selon index_train, et index_test
	train_seed, test = dataset.iloc[index_train, :], dataset.iloc[index_test, :] #train_test_split(dataset, test_size=(1/3), stratify=dataset["target"], random_state=42) 
	
	# Créer les fausses données, à partir de train_seed
	train = create_sample(train_seed, 1000)
	
	# Sauvegarder les données utiliser inférer les fausses données
	train_seed.to_csv("train_set_seed.csv", index=False)
	
	# Sauvegarder les données d'entraînement générées
	train.to_csv("train_set.csv", index=False)
	
	# Sauvegarder les données de test
	test.to_csv("test_set.csv", index=False)