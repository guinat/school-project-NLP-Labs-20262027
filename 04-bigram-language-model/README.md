# Transformers 1 - Modèle de langage bigramme

Modèle bigramme au niveau caractère entraîné sur Tiny Shakespeare, en suivant le début de la vidéo "Let's build GPT" de Karpathy. On passe du texte à des tokens entiers, on fait des mini-batchs de paires (contexte, cible), puis on entraîne une table 65 x 65 avec la cross-entropy.

## Lancer le TP

```bash
uv run main.py
```

- Environ 20 secondes sur CPU, avec les 10,000 steps d'entraînement.
- Le corpus est téléchargé dans `data/input.txt` au premier lancement.
- Les losses et les textes générés avant et après entraînement s'affichent dans le terminal.

## Résultats

Le corpus fait 1,115,394 caractères sur 40,000 lignes, avec 65 caractères différents :

```
"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
```

Les retours à la ligne font partie des données : chaque réplique commence par le nom du personnage suivi de deux-points, seul sur sa ligne (`First Citizen:`, `All:`), et les répliques sont séparées par une ligne vide. La ponctuation (`,` `.` `:` `;` `?` `!` `'`) est fréquente, et on garde la casse, donc `A` et `a` sont deux tokens différents. Le découpage 90/10 donne 1,003,854 tokens d'entraînement et 111,540 tokens de validation.

Le modèle est un seul `nn.Embedding(65, 65)` (4,225 paramètres), entraîné avec AdamW (lr 1e-3), un batch size de 4 et un block size de 8. Les losses train et val sont des moyennes sur 200 batchs tirés au hasard.

| Steps | Loss train | Loss val | Loss sur un batch |
|------:|-----------:|---------:|------------------:|
| 0 | 4.6475 | 4.6486 | 5.0364 |
| 100 | 4.5744 | 4.5477 | 4.4036 |
| 1,000 | 4.0036 | 4.0167 | 3.6823 |
| 2,000 | 3.5254 | 3.5706 | 3.4811 |
| 4,000 | 2.9901 | 2.9927 | 2.9578 |
| 6,000 | 2.7052 | 2.7248 | 2.6762 |
| 8,000 | 2.5768 | 2.6346 | 2.5141 |
| 10,000 | 2.5660 | 2.5768 | 2.5095 |

Au step 0, la loss sur un batch est celle du modèle non entraîné sur le batch d'exemple `xb`, `yb` affiché plus tôt. Ensuite, c'est la loss sur le dernier batch d'entraînement.

Un modèle qui prédit la distribution uniforme aurait une loss de -ln(1/65) = 4.1744. Le modèle non entraîné fait pire (4.65) parce que ses logits sont tirés selon N(0, 1) : chaque ligne de la table est une distribution aléatoire et pas uniforme, et avec ces logits la loss attendue est proche de ln(65) + 1/2 et pas de ln(65). La loss sur un seul batch (32 prédictions) est bruitée, c'est pour ça que le tableau donne aussi les estimations moyennées.

La loss se stabilise après quelques milliers de steps : la loss train baisse de 0.54 entre 2,000 et 4,000 steps, mais seulement de 0.01 entre 8,000 et 10,000. Comme référence, le script calcule aussi un bigramme en comptant les paires de caractères du train, avec un lissage add-one parce que certaines paires de la validation n'apparaissent jamais dans le train : 2.4546 en train et 2.4819 en val. Les comptes normalisés sont la table qui minimise la cross-entropy sur le train, donc c'est à peu près la loss la plus basse qu'un modèle bigramme peut atteindre, et la table entraînée finit environ 0.1 au-dessus. Les losses train et val restent proches à chaque étape : 4,225 paramètres ne peuvent pas mémoriser un million de caractères.

Chaque texte généré ci-dessous part du token 0 (`\n`) et tire 300 caractères avec la même graine pour le générateur aléatoire. Les tirages aléatoires sont donc identiques et seuls les poids changent d'un texte à l'autre.

Avant l'entraînement :

```
lpzuWaHytczde.QwEiT,YlrzlGu-s:eNYGiGbo
LzRjgkNU,-A.U bisV
OG bT,OPU.;,er$zyAnsz-imsz&AkfJf;ygqTRTNBETmyOm .VqVh -RLcJsl;HPX'vsqA&fgSr
lqGMR.E?AgqYptZJsA&Xn3MimKOSqIafvWtIgkgtrIRaZJ;3pM-
wH:HvLl;3v3IZZiEhOrXAklmsGsXzkNq!nzA&..3xaZ
OJe;;keA$?$gmn:e.Sl'NS?$OxPydtRpD;E?FUIqV
O:Ezr.SmrlM..BajzRCOw3M.otJQ
```

Après 100 steps, le texte est identique sauf à la quatrième ligne, où `ZJs` est devenu `:vd` :

```
lqGMR.E?AgqYpt:vdA&Xn3MimKOSqIafvWtIgkgtrIRaZJ;3pM-
```

Après 1,000 steps :

```
C$zlloSvTczde.QwEim,YlrzlGu-s:eNYGiGbo
LzRjgkNUSqARU bispJzG HR,OPU.;,erazyAnsz-imsz&AkfJf;ygqTRTNBETmyOm .VqVh MRLcJsl;HPX'vspA&fgSr
KZzyd.E?AgqYpt:vdA&Xn3MimKOSqIafvWtIgkgerIRaZJ;3pM-
wH:HvLl;3v3IZZig;OrXAklmsGsiumer nzA&..3xaZ
OJe;;keA$?$gmn:e.Sl'NS?$OxPydtRpD;E?FUIFV
O:Ezr.SmrlM..BauUZ:OhUCfotJQ
```

Après 10,000 steps :

```
C$zllon tcede.'aEin,Y rolound e waindo
LzRRSe t,-l.
THist ic wsostte;,

ROUMnsenimser se ay g prerancmy mou ber s Lcisl tem've tofr?n gnzyos IAg thoevere n:
BUK: deanver
Wt trerath;3pe t isevess tode ish rathoo bie med n
ARWingar
OUEinoes selmnsen l'CE?$OxPy the af, ty V: g;-h.
Thly a au a O:
F ote.
```

## Questions

### Étape 1 : corpus

*Quelle est l'unité d'un token ici ?*

Un caractère. Le modèle voit la pièce comme une suite plate de 1,115,394 caractères et n'a aucune notion de mot.

*Pourquoi passer `encoding="utf-8"` ?*

Sans ça, `open` utilise l'encodage par défaut de la plateforme (cp1252 sous Windows par exemple). Tiny Shakespeare est en ASCII pur, donc ici le résultat serait le même. Mais un caractère non ASCII (accents, guillemets typographiques) serait décodé autrement ou ferait une erreur avec un autre encodage par défaut. En le précisant, le vocabulaire, et donc les ids des tokens, sont les mêmes partout.

*Est-ce que le fichier est déjà découpé en exemples ?*

Non. C'est un seul flux continu de répliques, de noms et de lignes vides. Les exemples d'entraînement sont créés après, en découpant dedans des fenêtres aléatoires de `block_size + 1` caractères.

### Étape 2 : encodage et décodage

*Pourquoi trier `chars` ?*

Un `set` n'a pas d'ordre garanti, et le hachage des chaînes est aléatoire d'un processus Python à l'autre. Les ids pourraient donc changer d'un lancement à l'autre, et une table entraînée ne correspondrait plus à l'encodage. Trier fixe la correspondance : `\n` vaut 0, l'espace vaut 1, puis viennent la ponctuation (et le chiffre `3`), les majuscules et les minuscules.

*Que garantit le test `decode(encode(message)) == message` ?*

Que `stoi` et `itos` sont exactement inverses l'un de l'autre, donc passer par des entiers ne fait rien perdre (`"hii there"` devient `[46, 47, 47, 1, 58, 46, 43, 56, 43]` puis redevient le même texte). Le script fait la même vérification sur tout le corpus et elle renvoie aussi `True`. Ça marche seulement pour les caractères du vocabulaire : `encode("é")` lève une `KeyError`.

### Étape 3 : contextes et cibles

*Combien de prédictions donne ce bloc ?*

Huit. Une fenêtre de `block_size + 1 = 9` tokens donne 8 paires (contexte, cible) avec des contextes de longueur 1 à 8. Le modèle apprend donc aussi à prédire à partir de contextes courts, ce qui est nécessaire au début de la génération.

*Pourquoi ne pas mélanger train et validation ?*

La loss de validation doit être mesurée sur du texte que le modèle n'a jamais vu à l'entraînement, sinon elle mesure la mémorisation et pas la généralisation. On coupe le texte en deux morceaux contigus à 90% au lieu de mélanger les fenêtres, parce que des fenêtres qui se chevauchent mettraient des séquences presque identiques dans les deux ensembles.

*Pour t = 3, quels sont le contexte et la cible ?*

Contexte `[18, 47, 56, 57]` (`"Firs"`), cible `58` (`"t"`).

### Étape 4 : mini-batchs

*Pourquoi soustraire `block_size` à la borne du tirage ?*

Un indice de départ `i` doit vérifier `i + block_size + 1 <= len(data)` pour que `y = data[i + 1 : i + block_size + 1]` ait encore `block_size` éléments. `torch.randint(len(data) - block_size, ...)` renvoie au plus `len(data) - block_size - 1`, qui est le dernier départ valide. Sans ça, une fenêtre proche de la fin serait plus courte et `torch.stack` planterait.

*Qu'est-ce que B = 32 changerait ?*

Chaque step traiterait 32 fenêtres indépendantes, soit 256 prédictions au lieu de 32. Le gradient serait moyenné sur plus d'exemples, donc moins bruité (l'écart entre la loss sur un batch et la loss moyennée dans le tableau montre à quel point B = 4 est bruité), mais chaque step demanderait plus de calcul. Le modèle et son nombre de paramètres ne changent pas.

*Qu'est-ce que T = 64 changerait ?*

Chaque fenêtre donnerait 64 prédictions, avec des contextes jusqu'à 64 caractères. Pour le modèle bigramme, ça veut juste dire plus de prédictions par fenêtre, puisqu'il ne regarde que le caractère courant. Pour le GPT complet du TP 4 du cours (`07-gpt-from-scratch`), T est le contexte maximum que le modèle peut utiliser, il fixe la taille de la table d'embedding des positions, et le coût de l'attention augmente en T².

### Étape 5 : la table bigramme

*Pourquoi une table C x C ?*

Chacun des C = 65 caractères courants a sa ligne de 65 scores, un par caractère suivant possible. On lit directement l'embedding comme les logits, donc sa largeur doit être égale à la taille du vocabulaire. Après l'entraînement, une ligne est le log de la distribution (non normalisée) de ce qui suit ce caractère.

*Où vont apparaître d_emb et `lm_head` dans le TP 4 du cours ?*

Dans `07-gpt-from-scratch`, cette table est coupée en deux : `nn.Embedding(vocab_size, n_embd)` associe à chaque token un vecteur de taille d_emb = `n_embd`, les blocs Transformer travaillent dans cet espace, et `lm_head = nn.Linear(n_embd, vocab_size)` ramène le résultat à C logits. Ici, les deux matrices sont fusionnées en une seule table C x C, sans rien entre les deux.

### Étapes 6 et 7 : génération, entraînement et bilan

*Pourquoi garder seulement le dernier pas de temps pendant la génération ?*

Les logits à la position t prédisent le caractère à t + 1. Pendant la génération, toutes les positions sauf la dernière prédisent des caractères qui sont déjà dans la séquence. Seule la dernière donne la distribution du prochain caractère, qu'on ne connaît pas encore. Pour le modèle bigramme, `logits[:, -1, :]` dépend seulement de `idx[:, -1]`, mais passer toute la séquence permet de garder `generate` tel quel quand le modèle utilisera le contexte.

*Pourquoi aplatir B x T pour la loss ?*

`F.cross_entropy` attend des logits de forme `(N, C)` et des cibles de forme `(N,)`. Chaque position de chaque séquence est un problème de classification indépendant, donc les logits `(4, 8, 65)` deviennent 32 lignes de 65 scores, et la loss est la moyenne sur ces 32 prédictions.

*Comparer la loss et le texte avant l'entraînement, après 100 steps et après 1,000 steps.*

Avant l'entraînement, la loss vaut 4.65 et le texte est fait de caractères aléatoires, majuscules et ponctuation comprises. Après 100 steps, elle vaut 4.57 et seulement trois caractères du texte changent : AdamW avec un lr de 1e-3 bouge chaque poids d'environ 1e-3 par step, donc les logits bougent d'environ 0.1 au maximum, alors que l'écart-type de départ est de 1. Après 1,000 steps (4.00), 46 des 300 caractères générés sont différents du texte non entraîné, mais ça reste du bruit, avec au mieux une suite qui ressemble à un mot comme `iumer`. Après 10,000 steps (2.57), le texte a la forme de la pièce, avec des petits groupes qui ressemblent à des mots et des noms en majuscules suivis de deux-points (`BUK:`). Par contre, les seuls vrais mots sont des mots très courts et fréquents (`the`, `a`), puisque chaque caractère ne voit que le précédent.

### Objectifs du TP

*Pourquoi les tokens sont des entiers ?*

PyTorch travaille sur des nombres, et `nn.Embedding` prend des indices entiers. Un id n'a pas de sens tout seul : il sert juste à choisir une ligne de la table.

*Pourquoi x et y sont décalés de un ?*

La cible à la position t est le caractère qui la suit, donc `y[t] = x[t + 1]`.

*Que veulent dire B, T et C ?*

B est le batch size (le nombre de séquences indépendantes traitées en parallèle, 4 ici), T la dimension temporelle (les positions dans une séquence, jusqu'à `block_size = 8`), et C les channels. Dans ce TP, C est la taille du vocabulaire (65), puisque la sortie de l'embedding est utilisée directement comme logits.

*Comment les logits deviennent des probabilités, et qu'est-ce que l'entraînement change ?*

Un softmax sur la dernière dimension, `exp(logit) / sum(exp(logits))`, transforme chaque ligne de 65 scores en distribution, et `torch.multinomial` tire dedans. L'entraînement ne change que les 4,225 nombres de la table : le gradient de la cross-entropy augmente le score du caractère suivant observé et baisse les autres, ligne par ligne.
