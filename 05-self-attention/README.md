# Transformers 2 - Self-attention causale

TP basé sur la vidéo "Let's build GPT" de Karpathy. On part d'une simple moyenne des tokens précédents et on reconstruit la self-attention causale en quatre versions : avec des boucles, avec un produit matriciel, avec un softmax masqué, puis avec une tête d'attention dont les poids viennent des queries et des keys.

## Lancer le TP

```bash
uv run main.py
```

- Environ 3 secondes sur CPU, dont plus de la moitié pour importer torch.
- Pas de dataset : les entrées sont des tenseurs aléatoires avec des seeds fixes.
- Les résultats de chaque étape (shapes, poids `wei`, stabilité du softmax) s'affichent dans le terminal.

## Résultats

### Agrégation par produit matriciel

Avec `torch.manual_seed(42)`, `a` est la matrice triangulaire inférieure normalisée par ligne et `b` contient des entiers aléatoires :

```
a =
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
b =
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
c = a @ b =
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])
```

### De la moyenne causale à l'attention

Les versions 1 à 3 utilisent l'entrée jouet `x = torch.randn(4, 8, 2)` (seed 1337) et calculent, pour chaque position t, la moyenne de `x[:, :t+1]`.

| Version | Poids | Vérification |
|---|---|---|
| 1. boucles | `torch.mean(x[b, :t+1], dim=0)` | référence `xbow` |
| 2. produit matriciel | `tril(ones(T, T))` divisé par la somme de ses lignes | `allclose(xbow, xbow2)` : True |
| 3. softmax masqué | `zeros(T, T)`, `-inf` au-dessus de la diagonale, softmax | `allclose(xbow, xbow3)` : True |
| 4. tête d'attention | softmax de `q @ k^T / sqrt(head_size)` avec le masque causal | voir plus bas |

Premières lignes de la première séquence et sa moyenne cumulée :

```
x[0]                   xbow[0]
[ 0.1808, -0.0700]     [ 0.1808, -0.0700]
[-0.3596, -0.9152]     [-0.0894, -0.4926]
[ 0.6258,  0.0255]     [ 0.1490, -0.3199]
```

### Une tête d'attention

`x = torch.randn(4, 8, 32)` (seed 1337), `head_size = 16`, key, query et value sont des `nn.Linear(32, 16, bias=False)`.

```
x shape: (4, 8, 32)
q shape: (4, 8, 16)
k shape: (4, 8, 16)
wei shape: (4, 8, 8)
out shape: (4, 8, 16)
every row of wei sums to 1: True
weights on future tokens are all 0: True
```

Poids d'attention de la première séquence, `wei[0]` (ligne = position de la query, colonne = position de la key) :

```
1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
0.40  0.60  0.00  0.00  0.00  0.00  0.00  0.00
0.31  0.29  0.40  0.00  0.00  0.00  0.00  0.00
0.32  0.22  0.24  0.21  0.00  0.00  0.00  0.00
0.15  0.20  0.17  0.15  0.34  0.00  0.00  0.00
0.13  0.25  0.13  0.11  0.31  0.07  0.00  0.00
0.16  0.20  0.11  0.11  0.14  0.17  0.11  0.00
0.08  0.12  0.11  0.15  0.11  0.11  0.16  0.16
```

Le masque force toujours le triangle supérieur à 0, mais les lignes ne sont plus uniformes : dans les versions qui font une moyenne, la dernière ligne vaudrait 0.125 partout, ici elle va de 0.08 à 0.16. La tête n'est pas entraînée, donc ces préférences viennent seulement des projections initiales aléatoires.

### Stabilité du softmax

`q` et `k` aléatoires de variance 1, avec `torch.manual_seed(1337)` avant chaque head size :

| head_size | var(q @ k^T) | var(q @ k^T / sqrt(head_size)) |
|---|---|---|
| 4 | 4.50 | 1.13 |
| 16 | 18.44 | 1.15 |
| 64 | 55.90 | 0.87 |

Avec `z = [0.1, -0.2, 0.3, -0.2, 0.5]` :

```
softmax(z) = tensor([0.1925, 0.1426, 0.2351, 0.1426, 0.2872])
softmax(8 * z) = tensor([0.0326, 0.0030, 0.1615, 0.0030, 0.8000])
```

Chaque score est une somme de `head_size` produits de termes de variance 1, donc sa variance augmente comme `head_size`. Diviser par `sqrt(head_size)` la ramène proche de 1. Sans ça, le softmax sature : si on multiplie `z` par 8 (c'est `sqrt(64)`, l'échelle typique des scores non divisés pour une tête de taille 64), la plus grande valeur monte à 0.80 et les deux valeurs `-0.2` tombent à 0.003. À l'initialisation, chaque token ferait attention presque uniquement à une seule position et les gradients qui passent par les autres positions seraient proches de zéro.

## Shapes

- `B = 4` : taille du batch, nombre de séquences indépendantes traitées en parallèle.
- `T = 8` : dimension du temps (block size), nombre de positions dans chaque séquence.
- `C` : channels, taille du vecteur à chaque position (2 dans les étapes de moyenne, 32 pour la tête d'attention).
- `d_h = head_size = 16` : taille des vecteurs query, key et value de la tête.

Dans la tête, `x` est `(B, T, C)`, `q`, `k` et `v` sont `(B, T, d_h)`, `k.transpose(-2, -1)` est `(B, d_h, T)`, les scores `q @ k^T` et `wei` sont `(B, T, T)`, et `out = wei @ v` est `(B, T, d_h)`. `wei[b, t, s]` indique combien la position t prend de la position s dans la séquence b.

## Questions

*Étape 1 : calculer c_2 sans Python. Pourquoi la somme de chaque ligne de `a` vaut 1 ?*

Avec l'indexation à partir de 1 de la slide, `c_2 = a_21 b_1 + a_22 b_2 + a_23 b_3 = 0.5 * (2, 7) + 0.5 * (6, 4) + 0 * (6, 5) = (4, 5.5)`, c'est la deuxième ligne affichée plus haut (`c[1]` en Python). Chaque ligne de `tril(ones)` est divisée par sa propre somme, donc la ligne i (en comptant à partir de 1) contient i poids égaux à `1 / i`. `c_i` est alors la moyenne de `b_1 ... b_i` et reste à la même échelle que `b`.

*Étape 2 : quelle est la shape de `xprev` à t=0, t=3 et t=7 ? Pourquoi la moyenne est prise sur `dim=0` ?*

`(1, 2)`, `(4, 2)` et `(8, 2)` : `x[b, :t+1]` garde les t + 1 positions visibles et les C = 2 channels. Comme l'indice du batch est déjà fixé, la dim 0 de `xprev` est l'axe du temps, donc la moyenne sur cet axe donne un seul vecteur de dimension C, la shape de `xbow[b, t]`. Une moyenne sur la dim 1 ferait plutôt la moyenne des channels de chaque token.

*Étapes 3-4 : pourquoi `wei @ x` marche avec une matrice (T, T) et un tenseur (B, T, C) ? Que fait le softmax là où l'entrée vaut -inf ?*

`@` multiplie les deux dernières dimensions et fait du broadcast sur les autres, donc la matrice `(T, T)` est traitée comme `(B, T, T)` et chaque séquence fait `(T, T) @ (T, C) -> (T, C)`. On obtient `(B, T, C)` avec les mêmes poids pour toutes les séquences. Le softmax calcule `exp(-inf) = 0`, donc les positions masquées ont un poids exactement à 0 et les positions visibles sont normalisées entre elles. En partant de scores nuls, les t + 1 positions visibles ont chacune `1 / (t + 1)`, les mêmes poids que la matrice `tril` normalisée. C'est pour ça que `xbow3` correspond à `xbow`.

*Étape 5 : quelle ligne de `wei[0]` correspond au dernier token ? Pourquoi la sortie mélange `v` et pas directement `k` ?*

La dernière ligne (indice 7), car les lignes sont les positions des queries. C'est la seule ligne sans zéro, parce que le dernier token voit tout le contexte. `k` est la projection qu'on compare aux queries : elle décrit comment un token doit être trouvé. `v` est une projection à part pour ce que le token transmet une fois qu'il est sélectionné. Comme ça, un token peut être choisi pour une caractéristique et en apporter une autre.

*Bilan : vérifier la somme des lignes de `wei`, l'absence de poids vers le futur et la shape finale. Expliquer les rôles distincts de Q, K et V.*

Chaque ligne de `wei` a une somme de 1 (`allclose` avec des uns : True), `wei.triu(diagonal=1)` vaut exactement 0 sur tout le batch (True), et `out` a la shape `(4, 8, 16)`, c'est-à-dire `(B, T, d_h)`. La query du token t encode ce qu'il cherche, la key du token s encode ce que s contient, et leur produit scalaire mis à l'échelle donne l'affinité de t pour s. Après le masque et le softmax, ces affinités forment la ligne t de `wei`, qui fait la moyenne des values, c'est-à-dire l'information que chaque token envoie vraiment aux autres.
