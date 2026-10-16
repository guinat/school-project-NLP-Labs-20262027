# NLP 1 - Analyse de sentiment avec des modèles classiques

Classification des critiques de films Allociné (positif / négatif). On compare un Naive Bayes multinomial sur les comptages de mots avec une régression logistique sur du TF-IDF.

## Lancer le TP

```bash
uv run main.py
```

- Environ 4 min sur CPU.
- Le dataset (`tblard/allocine`, fichiers parquet) est téléchargé depuis Hugging Face au premier lancement, puis mis en cache.
- Les scores et les erreurs des modèles s'affichent dans le terminal. Tous les chiffres ci-dessous sont affichés par `main.py`.

## Préparation des données

Chaque split est nettoyé séparément. On supprime d'abord les lignes avec une valeur manquante, puis les doublons : deux critiques sont des doublons quand leurs textes normalisés sont identiques une fois la ponctuation et les espaces retirés. Ça attrape aussi les quasi-doublons comme "super film!!" et "super film!!!". On enlève aussi les critiques où il ne reste aucun mot (une critique ne contient qu'une URL). Si un texte est encore présent deux fois après ça, c'est qu'il a deux labels ("original" négatif et "original." positif dans test), donc on supprime les deux copies. Enfin, on retire de validation et de test les critiques qui sont aussi dans train, et on retire aussi de test les critiques de validation, puisque c'est validation qui sert à choisir les modèles.

| split | brut | dédoublonné | gardé | part de positifs |
|---|---|---|---|---|
| train | 160000 | 159297 | 159297 | 0.5039 |
| validation | 20000 | 19941 | 19890 | 0.4898 |
| test | 20000 | 19920 | 19882 | 0.4801 |

Le filtre anti-fuite retire 51 critiques de validation et 38 de test. Les classes sont équilibrées, et le test garde sa distribution d'origine (48% de positifs).

## Prétraitement

On applique la même normalisation avant les deux modèles :

- On décode les entités HTML et on met le texte en minuscules. Les majuscules peuvent marquer l'insistance ("EXCELLENT"), mais les garder découperait les mots en variantes de casse plus rares.
- Les URL et les balises HTML sont remplacées par un espace. Les deux sont rares (788 critiques brutes contiennent une URL, 4 contiennent des balises comme `<br>` ou `<p>`). Le motif des balises liste leurs noms, parce qu'un `<[^>]+>` générique supprimerait aussi les citations écrites `<< ... >`.
- Les espaces multiples sont réduits à un seul.
- Les tokens sont les mots, `!`, `?` et les smileys simples (`\b\w+\b|[!?]|[:;]-?[()]`). Le motif par défaut de scikit-learn supprime ces trois-là, et aussi les tokens d'une seule lettre comme le "n" de "n'est". Les smileys comme `:)` sont rares mais apparaissent surtout dans les critiques positives.
- Pas de suppression des stopwords, pas de stemming, pas de retrait des accents. "pas", "ne" et "mais" sont dans la plupart des listes de stopwords françaises, et les enlever détruit les négations et les contrastes.
- Les n-grams vus dans une seule critique de train sont ignorés (`min_df=2`) : 85972 features en unigrammes, 859830 avec les bigrammes.

Le vectorizer fait partie de chaque pipeline, donc son vocabulaire et ses IDF sont appris sur train uniquement.

## Protocole

Pour chaque modèle, on choisit la plage de n-grams et l'hyperparamètre principal sur validation, avec le F1 macro (seuil 0.5). Ensuite on choisit aussi sur validation le seuil de décision sur la probabilité positive, puis on évalue le pipeline final une seule fois sur test.

F1 macro sur validation, comptages + Naive Bayes :

| ngram_range | alpha=0.1 | alpha=0.3 | alpha=1.0 | alpha=3.0 |
|---|---|---|---|---|
| (1, 1) | 0.8914 | 0.8928 | 0.8938 | 0.8938 |
| (1, 2) | 0.9212 | 0.9219 | **0.9221** | 0.9211 |

F1 macro sur validation, TF-IDF (`sublinear_tf=True`) + régression logistique (liblinear) :

| ngram_range | C=1 | C=3 | C=10 | C=30 |
|---|---|---|---|---|
| (1, 1) | 0.9261 | 0.9280 | 0.9270 | 0.9200 |
| (1, 2) | 0.9395 | 0.9437 | **0.9458** | 0.9449 |

Les bigrammes donnent le plus gros gain pour les deux modèles (+0.028 pour Naive Bayes, +0.018 pour la régression logistique). Le paramètre de lissage du Naive Bayes ne change presque rien : avec les bigrammes, alpha de 0.1 à 3 reste entre 0.9211 et 0.9221.

F1 macro sur validation en fonction du seuil de décision, pour les réglages retenus :

| seuil | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 | 0.55 | 0.6 | 0.65 | 0.7 |
|---|---|---|---|---|---|---|---|---|---|
| Naive Bayes | 0.9220 | 0.9219 | 0.9222 | 0.9221 | 0.9221 | **0.9225** | 0.9216 | 0.9214 | 0.9208 |
| régression logistique | 0.9334 | 0.9381 | 0.9413 | 0.9439 | 0.9458 | **0.9464** | 0.9448 | 0.9429 | 0.9381 |

Les deux modèles finissent avec 0.55, un petit gain par rapport à 0.5 (0.0004 pour Naive Bayes, 0.0006 pour la régression logistique). La courbe du Naive Bayes est presque plate, ce qui colle avec ses probabilités extrêmes (ses erreurs faites avec le plus de confiance ont toutes p=1.0000).

## Résultats

Jeu de test (19882 critiques). Le temps de fit comprend la vectorisation de train et l'entraînement du classifieur. Le temps de prédiction comprend la vectorisation et la prédiction sur tout le test.

| | comptages + Naive Bayes | TF-IDF + régression logistique | classe majoritaire |
|---|---|---|---|
| F1 macro validation | 0.9225 | 0.9464 | 0.3288 |
| seuil de décision | 0.55 | 0.55 | - |
| accuracy | 0.9223 | **0.9453** | 0.4801 |
| précision négatif | 0.9180 | 0.9429 | 0.0000 |
| rappel négatif | 0.9341 | 0.9524 | 0.0000 |
| F1 négatif | 0.9260 | 0.9476 | 0.0000 |
| précision positif | 0.9273 | 0.9479 | 0.4801 |
| rappel positif | 0.9096 | 0.9376 | 1.0000 |
| F1 positif | 0.9183 | 0.9427 | 0.6487 |
| F1 macro | 0.9222 | **0.9452** | 0.3244 |
| temps de fit (s) | 22.2 | 47.5 | - |
| temps de prédiction (s) | 2.3 | 2.4 | - |

Matrices de confusion sur test (lignes : vrai label, colonnes : prédiction) :

| comptages + Naive Bayes | prédit négatif | prédit positif |
|---|---|---|
| vrai négatif | 9656 | 681 |
| vrai positif | 863 | 8682 |

| TF-IDF + régression logistique | prédit négatif | prédit positif |
|---|---|---|
| vrai négatif | 9845 | 492 |
| vrai positif | 596 | 8949 |

La régression logistique fait 1088 erreurs contre 1544 pour Naive Bayes, et 710 critiques de test sont mal classées par les deux. Les deux modèles ratent plus de critiques positives que négatives (863 contre 681 pour Naive Bayes, 596 contre 492 pour la régression logistique). Pour la régression logistique, c'est en partie l'effet du seuil à 0.55. Le F1 macro de validation et celui de test diffèrent de moins de 0.002 pour les deux modèles, donc les petites grilles n'ont pas fait d'overfitting sur validation.

La baseline majoritaire prédit toujours "positif" (la classe majoritaire de train), donc son accuracy est la part de positifs du test (0.4801) et son F1 négatif vaut 0.

Construire les features unigrammes + bigrammes sur train prend 22.1 s pour les comptages et 27.5 s pour le TF-IDF. Le fit liblinear ajoute environ 20 s, le fit du Naive Bayes presque rien. La prédiction prend 2.3 s (Naive Bayes) et 2.4 s (régression logistique) pour les 19882 critiques de test.

Plus grands poids de la régression logistique :

```
most negative: ennuyeux, mauvais, navet, ennui, intérêt, déception, plv, nul, décevant, ridicule, raté, malheureusement, médiocre, plat, rien, pas, mal, daube, pas terrible, nanar
most positive: excellent, magnifique, superbe, bon film, très bon, pas mal, ennuie pas, beau film, parfaitement, sublime, parfait, excellente, génial, remarquable, excellent film, merveille, émouvant, adoré, a voir, excellents
```

"pas mal", "ennuie pas" et "pas terrible" sont des négations qui n'existent qu'en bigrammes : "mal" et "pas" seuls sont négatifs, alors que "pas mal" est une des features les plus positives. "intérêt" est négatif, sûrement à cause d'expressions comme "aucun intérêt" ou "sans intérêt". "plv" n'est pas un mot : certaines critiques finissent par une ligne `PLV : <short verdict>`, apparemment la signature d'un seul auteur, et presque toutes sont labellisées négatives. Le modèle apprend l'auteur plutôt que le sentiment, et ça ne marche ici que parce que cet auteur apparaît dans les trois splits.

## Analyse des erreurs

Les cinq critiques de test mal classées par la régression logistique avec la probabilité prédite la plus haute (`main.py` affiche les textes complets, les longs sont raccourcis ici) :

1. Vrai positif, prédit négatif (p=0.9997).
   > Un film si mauvais que ca en devient cosmique. Tout est raté dans ce film complêtement débil, les dialogues sont mauvais, la mise en scène est affligeante, [...] les auteurs ont essayé de filmer avec les pieds un projet qu'on oserait pas écrire sur du pq de peur de se salir l'anus, en ce sens ils ont réussis leur pari.

   Toutes les phrases sont négatives. Seuls "cosmique" et la dernière proposition font penser à une note du genre "tellement nul que c'est bien".

2. Vrai négatif, prédit positif (p=0.9997).
   > Grosse fatigue est une bonne comédie.Le scénario est original,prenant et bien construit. [...] Grosse fatigue un donc un bon film drôle,divertissant,jubilatoire et méditatif(la fin du film et son constat pessimiste et vériste de l'évolution du cinéma français).

   Le texte est positif du début à la fin, donc le label est sûrement faux et le modèle a raison sur le texte.

3. Vrai négatif, prédit positif (p=0.9994).
   > Vraiment bon!

   Soit le label est faux, soit c'est du sarcasme sans aucun indice dans le texte.

4. Vrai positif, prédit négatif (p=0.9992).
   > J'ai été très déçu par "Les fils de l'Homme", car il avait tout pour être bon : un bon réalisateur, des bons acteurs [...] une bonne idée de départ [...] Mais cette bonne idée de départ va vite dégénérer dans un grand n'importe quoi! [...] Désolé pour ceux qui ont aimé, mais pour moi ce film est plus que décevant...

   Une structure mixte avec "mais" : les mots positifs décrivent ce que le film aurait pu être, et le verdict après "Mais" est négatif. Le modèle lit correctement le texte malgré les nombreux "bon" et "bonne", mais le label dit positif.

5. Vrai négatif, prédit positif (p=0.9988).
   > Mon dieu, quel beau film! Plein de poésie, intelligent, c'est dommage qu'il n'y a plus de film comme celui-ci. A voir absolument! (ceci est un fake)

   Le label est juste : la critique est ironique, et seul le "(ceci est un fake)" final le dit. "beau film", "a voir" et "!" sont des features positives fortes, et le seul mot "fake" ne suffit pas à les compenser.

Trois des cinq (2, 3, 4) ressemblent à du bruit dans les labels ou à une note qui contredit le texte, les deux autres (1, 5) sont de l'ironie. Une partie des erreurs restantes ne peut pas être corrigée par un meilleur classifieur.

Les cinq erreurs du Naive Bayes faites avec le plus de confiance sont différentes : elles ont toutes p=1.0000 et ce sont toutes des critiques longues et nuancées. La critique de "Géant" fait longuement l'éloge des acteurs avant de conclure négativement ("très bonne,mais pas à ce point-là", "En dépit de tous ces atouts [...] laisse un goût de laisser-aller"). Celle du documentaire de Vik Muniz finit par une expression positive niée ("Pas le chef d’œuvre attendu"). La critique positive de "Les Bronzés font du ski" passe plusieurs phrases à dire à quel point le premier film l'avait déçu ("m'avait relativement déçu"). Le Naive Bayes multiplie une vraisemblance par token avec l'hypothèse d'indépendance, donc beaucoup de mots d'une même polarité poussent la probabilité à 1, quel que soit le verdict final.

## Quel modèle garder

TF-IDF avec unigrammes et bigrammes + régression logistique (C=10, seuil 0.55) : 2.3 points de F1 macro au-dessus du Naive Bayes (0.9452 contre 0.9222) et des poids lisibles, pour environ deux fois le temps de fit (47.5 s contre 22.2 s). Le Naive Bayes reste une baseline peu coûteuse.
