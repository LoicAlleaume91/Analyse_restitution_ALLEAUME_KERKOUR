
# Analyse intelligente des avis clients McDonald's

**Sujet 5 : Analyse intelligente des avis clients (NLP)**

Master 1 Machine Learning pour la Science des Données  
Université Paris Cité  
Module : Analyse et restitution  
Année universitaire 2026-2027

**Équipe :**
- Dina Kerkour, membre A
- Loïc Alleaume, membre B

**Phase actuelle :** Cadrage et exploration initiale

## 1. Contexte du projet

Les avis laissés par les clients constituent une source d'information intéressante pour les entreprises. Ils permettent de recueillir des retours sur les produits, les services et l'expérience vécue dans les établissements.

Cependant, une note ne suffit pas toujours à expliquer la satisfaction ou l'insatisfaction d'un client. Deux personnes peuvent attribuer la même note à un restaurant pour des raisons très différentes. Les commentaires apportent davantage de détails, mais leur analyse devient difficile lorsqu'ils sont nombreux.

Nous avons choisi de travailler sur des avis de restaurants McDonald's pour étudier comment les techniques de traitement automatique du langage naturel (NLP) peuvent aider à exploiter ces informations.

Notre démarche ne consiste pas uniquement à prédire un sentiment. Nous souhaitons également comprendre les sujets abordés dans les commentaires et réfléchir à la manière de présenter ces analyses sous une forme utile à la prise de décision.

## 2. Problématique

**Comment transformer les avis textuels des clients de McDonald's en indicateurs décisionnels fiables ?**

Cette problématique constitue le point de départ de notre projet.

Pour y répondre, nous nous intéresserons à plusieurs aspects.

### Comprendre la satisfaction des clients

Nous chercherons à déterminer si le contenu d'un commentaire permet de prédire un sentiment positif, neutre ou négatif.

Les notes attribuées serviront à construire les étiquettes nécessaires à l'apprentissage supervisé, tandis que les textes constitueront les données analysées par les modèles.

### Comprendre les raisons derrière les avis

Une classification positive ou négative ne permet pas, à elle seule, de savoir ce qui a influencé l'expérience du client.

Nous chercherons donc à identifier les sujets récurrents dans les commentaires afin de mieux comprendre les principaux motifs de satisfaction et d'insatisfaction.

### Évaluer la fiabilité des analyses

Les indicateurs produits dépendront directement de la qualité des données et des méthodes utilisées.

Nous étudierons notamment les performances des modèles, leurs erreurs, l'effet du nettoyage des textes et la cohérence des thèmes identifiés.

Nous comparerons également les gains apportés par les Transformers avec leur coût de calcul.

### Rendre les résultats exploitables

Enfin, nous souhaitons présenter les résultats de manière compréhensible, notamment à travers des visualisations et un tableau de bord interactif.

L'objectif sera de faire ressortir les informations qui pourraient aider à identifier des points d'amélioration dans l'expérience client.

## 3. Objectifs

Notre travail s'organise autour de cinq objectifs principaux.

1. **Explorer et préparer les données.** Examiner la structure du dataset, sa qualité, les valeurs manquantes, les doublons et la répartition des notes.

2. **Classifier les sentiments.** Construire et comparer plusieurs modèles capables d'identifier les sentiments positifs, neutres et négatifs dans les commentaires.

3. **Étudier les choix de préparation.** Comparer différentes configurations de nettoyage et examiner leur influence sur les résultats.

4. **Identifier les thèmes récurrents.** Utiliser des méthodes d'analyse thématique et d'extraction de mots-clés pour mieux comprendre les sujets abordés par les clients.

5. **Évaluer et restituer les résultats.** Mesurer la fiabilité et les coûts des méthodes utilisées, puis présenter les principaux constats dans un tableau de bord et une note de synthèse.

## 4. Périmètre et données

### 4.1. Dataset retenu

Nous avons choisi le fichier `McDonald_s_Reviews.csv`, proposé dans le cadre du sujet.

Le dataset initial contient :

- **33 396 avis clients**
- **10 variables**
- **40 établissements McDonald's**
- **Catégorie :** Fast food restaurant
- **Localisation :** États-Unis
- **Données principales :** commentaires, notes et informations sur les restaurants

Ces informations correspondent au fichier initial et seront vérifiées lors de l'exploration des données.


### 4.2. Variables du dataset

Le dataset contient 10 variables :

- **`reviewer_id`** : identifiant du client ayant laissé l'avis.
- **`store_name`** : nom du restaurant.
- **`category`** : catégorie de l'établissement.
- **`store_address`** : adresse du restaurant.
- **`latitude`** : latitude de l'établissement.
- **`longitude`** : longitude de l'établissement.
- **`rating_count`** : nombre d'évaluations associé à l'établissement.
- **`review_time`** : ancienneté relative de l'avis.
- **`review`** : commentaire rédigé par le client.
- **`rating`** : note attribuée par le client.

Les variables `review` et `rating` seront principalement utilisées pour la classification des sentiments. Les autres variables pourront servir à l'exploration des données et à la restitution des résultats, selon leur pertinence.
### 4.3. Construction des sentiments

Conformément au cahier des charges, les notes seront regroupées en trois classes :

- **1 ou 2 étoiles :** sentiment négatif.
- **3 étoiles :** sentiment neutre.
- **4 ou 5 étoiles :** sentiment positif.

Les modèles de classification utiliseront le texte des commentaires pour apprendre à prédire ces catégories.

Cependant, les notes ne reflètent pas toujours parfaitement le sentiment exprimé dans les commentaires. Cette limite sera prise en compte lors de l'analyse des résultats.
### 4.4. Points de vigilance

Plusieurs éléments devront être examinés avant les expérimentations :

- Les doublons et les commentaires identiques.
- Les commentaires vides, très courts ou peu informatifs.
- La langue des textes.
- Le déséquilibre entre les classes de sentiments.
- Les incohérences possibles entre les notes et les commentaires.
- Les limites des informations temporelles disponibles.

La préparation des données devra également éviter les fuites d'information entre les ensembles d'entraînement, de validation et de test.

## 5. Niveau et démarche envisagée

Nous prévoyons de suivre les différentes étapes du cahier des charges, en adaptant nos choix aux caractéristiques du dataset et aux ressources disponibles.

### 5.1. Classification des sentiments

Nous commencerons par des modèles classiques utilisant TF-IDF pour représenter les textes.

Les méthodes envisagées comprennent :

- La régression logistique.
- Les SVM.
- Naive Bayes.

Nous étudierons ensuite des modèles Transformers, notamment DistilBERT et BERT, ou une architecture adaptée à la langue des commentaires.

Les comparaisons porteront principalement sur le F1-score macro, les performances par classe, les matrices de confusion, le temps d'inférence et la consommation mémoire.

### 5.2. Analyse des thèmes

Pour identifier les sujets abordés dans les commentaires, plusieurs méthodes pourront être expérimentées :

- LDA.
- Clustering sur des représentations textuelles.
- BERTopic.
- Extraction de mots-clés avec TF-IDF, KeyBERT ou d'autres méthodes adaptées.

Les thèmes obtenus devront être examinés et interprétés à partir des commentaires.

### 5.3. Évaluation humaine

Une partie de l'évaluation reposera sur l'annotation indépendante de 100 avis par les deux membres du binôme.

Nous chercherons à comparer les annotations humaines, à mesurer leur accord et à examiner les éventuelles divergences avec les résultats des modèles.

### 5.4. Restitution

Nous envisageons de construire un tableau de bord interactif pour présenter les sentiments, les thèmes dominants et les principaux constats.

Selon les résultats de l'exploration, des comparaisons entre établissements ou des représentations géographiques pourront également être étudiées.

Une note de synthèse accompagnera les analyses pour présenter les conclusions et les recommandations de manière accessible.

## 6. Langages et outils envisagés

Nous envisageons d'utiliser R et Python, en fonction des besoins de chaque partie du projet.

Le choix des outils définitifs sera fait progressivement, en tenant compte de leur pertinence, de leur facilité d'utilisation et de leur compatibilité.

| Besoin | Outils possibles |
|---|---|
| Exploration et statistiques | R, Python |
| Manipulation des données | dplyr, tidyr, Pandas |
| Visualisation | ggplot2, Matplotlib, Seaborn |
| Classification classique | tidymodels, scikit-learn |
| Transformers | Python, Hugging Face, PyTorch |
| Analyse des thèmes | Packages R, gensim, BERTopic |
| Environnement de travail | RStudio, Jupyter, Google Colab |
| Tableau de bord | R Shiny, Streamlit, Dash, Power BI |
| Versionnement | Git, GitHub |
| Suivi du projet | Jira |

L'utilisation de deux langages nécessitera de définir des formats communs pour échanger les données et les résultats entre les différentes étapes.

## 7. Répartition du travail

Nous avons réparti les responsabilités en deux parties complémentaires, conformément aux indications du cahier des charges.

### Membre A : Dina Kerkour

**Classification des sentiments**

Responsabilités principales :

- Préparer les étiquettes de sentiment.
- Préparer les données pour la classification.
- Définir un découpage reproductible des données.
- Étudier l'effet du nettoyage et du déséquilibre des classes.
- Entraîner les modèles classiques et les Transformers.
- Comparer les performances et les coûts de calcul.
- Analyser les erreurs de prédiction.

### Membre B : Loïc Alleaume

**Analyse des thèmes et restitution**

Responsabilités principales :

- Réaliser l'exploration initiale des données.
- Préparer les textes pour l'analyse thématique.
- Expérimenter les méthodes de découverte de thèmes.
- Extraire et interpréter les mots-clés.
- Évaluer la cohérence des thèmes.
- Développer le tableau de bord.
- Préparer la note de synthèse.

### Travail commun

Certaines étapes nécessiteront une collaboration directe :

- Définition du périmètre et des choix méthodologiques.
- Validation des règles de préparation des données.
- Annotation indépendante de 100 avis.
- Analyse et interprétation des résultats.
- Rédaction du rapport.
- Préparation de la présentation finale.

La répartition des tâches permet à chacun d'avancer sur son périmètre, tout en conservant une compréhension commune du projet.

## 8. Organisation du dépôt et suivi

Nous utiliserons GitHub pour centraliser les fichiers et suivre les modifications.

Loïc avait initialement proposé une organisation par étapes du projet, avec des dossiers consacrés aux différentes analyses. Cette première proposition a ensuite été remplacée par une nouvelle arborescence proposée par Dina, permettant de mieux organiser les fichiers selon leur fonction.

La nouvelle organisation prévoit notamment les dossiers suivants :

- **`configs/`** pour les paramètres de configuration et de nettoyage.
- **`data/`** pour les données brutes et préparées.
- **`notebooks/`** pour les notebooks d'exploration et d'analyse.
- **`src/`** pour les scripts et fonctions réutilisables.
- **`results/`** pour les résultats, graphiques et tableaux.
- **`dashboard/`** pour le tableau de bord interactif.
- **`docs/`** pour les documents et livrables du projet.
- **`tests/`** pour vérifier le fonctionnement du code.

Cette structure constitue l'organisation cible du dépôt. Les dossiers seront créés progressivement selon les besoins du projet.

Les développements seront réalisés sur des branches dédiées, puis relus avant leur intégration.

Nous utiliserons également :

- **Jira** pour suivre les tâches et les responsabilités.
- **`CONTRIBUTIONS.md`** pour documenter le travail réalisé par chaque membre.
- **`results/experiments.csv`** pour conserver une trace des expériences et des résultats.

Un point de synchronisation hebdomadaire permettra de partager notre avancement et de coordonner les étapes communes.

## 9. Planning général

| Période | Travail prévu |
|---|---|
| Novembre 2026 | Cadrage, exploration, préparation des données et premières baselines |
| Novembre à décembre 2026 | Nettoyage paramétrable, DistilBERT et premiers thèmes LDA |
| 5 janvier 2027 | Présentation des premiers résultats |
| Janvier à février 2027 | Expérimentations complémentaires, évaluation humaine et tableau de bord |
| 1er mars 2027 | Revue des résultats et corrections |
| 15 mars 2027 | Soutenance finale |

## 10. Avancement initial

**Séance 1 : Lancement du projet, 10 novembre 2026**

Travaux engagés :

- [x] Choix du sujet.
- [x] Choix du dataset McDonald's.
- [x] Répartition des responsabilités.
- [x] Premières propositions de problématique et d'objectifs.
- [x] Mise en place du suivi des tâches sur Jira.
- [x] Début de l'exploration des données sous R Markdown.
- [ ] Validation commune du README.
- [ ] Confirmation du niveau du projet.
- [ ] Première baseline TF-IDF.

Cette première version du README présente notre cadrage initial. Elle sera complétée au fil du projet avec les choix techniques retenus, les résultats expérimentaux et les instructions nécessaires à la reproduction des analyses.
