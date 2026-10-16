# Transformers 3 - LayerNorm, feed-forward et connexion résiduelle

On code une LayerNorm à la main et on la compare à `nn.LayerNorm`. Ensuite on construit le feed-forward (C -> 4C -> C) et le bloc pre-norm `x + ffwd(ln(x))` qui sert plus tard dans le modèle GPT.

## Lancer le TP

```bash
uv run main.py
```

- Environ 1.3 s sur CPU, surtout pour importer torch.
- Pas de données ni d'entraînement, tout tourne sur des tenseurs aléatoires.
- Les résultats s'affichent dans le terminal.

## Résultats

### LayerNorm

`x = torch.randn(32, 100)` (32 tokens, 100 features), `y = LayerNorm1d(100)(x)` :

| Sélection | Ce qui varie | Moyenne | Écart-type |
|---|---|---|---|
| `y[:, 0]` | les 32 tokens, feature 0 | 0.1476 | 0.8847 |
| `y[0, :]` | les 100 features du token 0 | -0.0000 | 1.0050 |

La plus grande moyenne par token vaut 3.1e-08 en valeur absolue, alors que les moyennes par feature montent jusqu'à 0.594. L'écart-type de 1.0050 au lieu de 1 vient de `torch.std`, qui utilise l'estimateur non biaisé. Les lignes sont normalisées avec la variance biaisée, donc leur écart-type non biaisé vaut sqrt(100/99) = 1.0050.

`torch.allclose(module(x), nn.LayerNorm(100)(x), atol=1e-6)` renvoie `True` (écart max 4.77e-07). `nn.LayerNorm` démarre avec weight à 1 et bias à 0, les mêmes valeurs que `gamma` et `beta`, donc les deux sorties correspondent sans copier aucun paramètre. Le test passe aussi sur un tenseur (4, 8, 32), car les statistiques n'utilisent que la dernière dimension.

### Feed-forward

`FeedForward(100)` sur une entrée (32, 100) donne des activations cachées de forme (32, 400) et un `delta` de forme (32, 100). Après la ReLU, 0.507 des activations cachées sont positives.

### Variantes du bloc pre-norm

Bloc : `nn.LayerNorm(32)` puis `FeedForward(32)`, avec B, T, C = 4, 8, 32. `x`, `delta` et `x_out = x + delta` ont tous la forme (4, 8, 32).

ReLU contre GELU, mêmes poids, entrée `ln(x)` :

| Activation | Positives | Non nulles |
|---|---|---|
| ReLU | 0.502 | 0.502 |
| GELU | 0.502 | 1.000 |

GELU(z) = z * Phi(z), avec Phi la fonction de répartition de la loi normale centrée réduite, est positive exactement quand z l'est, donc la part de valeurs positives ne change pas. La différence est du côté négatif : GELU renvoie de petites valeurs négatives au lieu de 0. Aucune unité cachée n'est donc complètement éteinte, et les entrées négatives ont encore un gradient.

Taille cachée et nombre de paramètres :

| h caché | Paramètres | 2Ch + h + C | Sortie |
|---|---|---|---|
| 64 (2C) | 4192 | 4192 | (4, 8, 32) |
| 128 (4C) | 8352 | 8352 | (4, 8, 32) |
| 256 (8C) | 16672 | 16672 | (4, 8, 32) |

On a Ch + h paramètres pour la première couche (poids et biais), plus hC + C pour la seconde. Ça fait (2C + 1)h + C, donc doubler h double à peu près le nombre de paramètres. La forme de sortie ne dépend pas de h. GELU n'a pas de paramètres, donc le tableau est le même avec les deux activations, et la LayerNorm ajoute 2C = 64 (gamma et beta).

Sans l'addition résiduelle :

| Sortie | Forme | cos(x, out) | \|out - x\| / \|x\| |
|---|---|---|---|
| `x + delta` | (4, 8, 32) | 0.969 | 0.250 |
| `delta` seul | (4, 8, 32) | 0.023 | 1.024 |

Les formes sont identiques, donc enlever l'addition ne lève aucune erreur.

Mélange des tokens : si on remplace le token 3 dans chaque séquence, `delta` ne change qu'à la position 3.

```
max change of delta per position after editing token 3:
0.0000 0.0000 0.0000 0.6496 0.0000 0.0000 0.0000 0.0000
```

## Questions

*Sur quel axe la LayerNorm doit-elle normaliser ?*

Le dernier (`dim=-1`), c'est-à-dire les C features d'un token. Chaque vecteur de token est centré et mis à l'échelle tout seul, indépendamment des autres positions et du batch. La BatchNorm fait l'inverse : elle normalise chaque feature sur le batch.

*À quoi servent eps, gamma et beta ?*

eps évite que le dénominateur soit nul. Un token constant a une variance nulle : avec eps=0 la sortie vaut `nan` (0/0), alors qu'avec eps=1e-5 on obtient 0. gamma et beta sont une échelle et un décalage appris par feature, initialisés à 1 et 0. Ils permettent au modèle de redonner à chaque feature la moyenne et l'étalement qui conviennent à la couche suivante, donc la normalisation ne limite pas ce que le bloc peut représenter.

*Quel print donne à peu près (0, 1) ?*

`y[0, :]` (moyenne -0.0000, écart-type 1.0050), parce qu'il parcourt les 100 features d'un token, ce qui est exactement l'axe normalisé. `y[:, 0]` suit une feature sur les 32 tokens, et la LayerNorm ne contraint pas ça : moyenne 0.1476, écart-type 0.8847.

*Quelle proportion des activations est positive ?*

À peu près la moitié : 0.507 sur l'entrée aléatoire (32, 100) et 0.502 sur l'entrée normalisée (4, 8, 32). Avec une entrée de moyenne nulle et des poids initialisés au hasard, les pré-activations sont à peu près symétriques autour de 0. La ReLU met donc à zéro environ la moitié des 4C unités cachées. Ça vient de l'initialisation, pas de quelque chose d'appris.

*Pourquoi revenir à C ?*

La sortie `delta` est ajoutée à `x` dans la connexion résiduelle, donc elle doit avoir la même forme (B, T, C). Le bloc suivant et la projection finale vers le vocabulaire attendent aussi des vecteurs de taille C. La couche 4C est juste un espace interne où la non-linéarité sélectionne des features.

*Quelle dimension la LayerNorm préserve-t-elle ?*

Elle garde la forme : (B, T, C) en entrée, (B, T, C) en sortie. Elle ne fait que centrer et remettre à l'échelle les C composantes de chaque token (puis appliquer gamma et beta). La taille d'embedding C et les positions des tokens restent donc les mêmes.

*Est-ce que le MLP mélange les tokens ?*

Non. `nn.Linear` agit sur la dernière dimension, donc les mêmes poids sont appliqués à chaque position séparément : modifier le token 3 ne change que la position 3 (voir la sortie plus haut). La LayerNorm ne mélange pas les positions non plus, c'est le rôle de l'attention.

*À quoi sert la connexion résiduelle ?*

Le bloc calcule une correction qui est ajoutée à `x` au lieu de le remplacer. La sortie garde une similarité cosinus de 0.969 avec l'entrée et bouge de 25% de sa norme, alors que sans l'addition elle n'a presque plus de lien avec l'entrée (cosinus 0.023). Dans un empilement profond, le chemin identité ramène aussi le gradient directement jusqu'aux premières couches, et c'est ce qui permet d'entraîner beaucoup de blocs.
