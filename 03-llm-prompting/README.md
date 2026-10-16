# NLP 3 - Prompting d'un petit LLM

On fait tourner Qwen3-0.6B en local sur CPU et on teste des prompts sur le format, la longueur, la langue, les faits, le raisonnement (thinking on / off), le résumé, les hallucinations et le sentiment (zero-shot et few-shot). Les sorties sont évaluées à la main (pertinence, cohérence, respect des consignes), avec un check automatique par prompt quand c'est possible.

## Lancer le TP

```bash
uv run main.py
```

- Environ 7 min sur CPU.
- C'est un script Python simple, pas le notebook Colab proposé par l'énoncé.
- Le modèle (`Qwen/Qwen3-0.6B`) est téléchargé dans le cache Hugging Face au premier lancement.
- Les prompts, les sorties, les temps et les checks sont écrits dans `results/generations.md`.
- Pic mémoire (max RSS) : 4.6 GiB.

## Modèle et décodage

- Modèle : `Qwen/Qwen3-0.6B` (0.6B paramètres, 28 couches), chargé avec `AutoModelForCausalLM`. Les prompts sont des listes de messages `system` / `user` / `assistant`, transformées en texte par le chat template du tokenizer.
- Précision : bfloat16, fixée dans le code. Le test de vitesse au début de `main.py` génère 64 tokens en greedy avec chaque précision : 6.19 tokens/s en float32 et 6.48 en bfloat16, donc seulement 5% plus rapide. Lire deux fois moins d'octets par token aide un peu, mais ce CPU n'a pas de calcul bfloat16 natif.
- Threads : 6. Le CPU (Intel i5-1335U) a 2 cœurs performance et 8 cœurs efficacité, et la génération avec les 12 threads était beaucoup plus lente.
- Thinking désactivé (`enable_thinking=False`) pour tous les prompts sauf deux : décodage greedy, donc les sorties sont les mêmes à chaque lancement. Au maximum 256 nouveaux tokens, 8 pour les labels de sentiment.
- Thinking activé : sampling avec les réglages conseillés dans la model card (température 0.6, top-p 0.95, top-k 20), car le décodage greedy a tendance à boucler dans ce mode, avec `torch.manual_seed(1337)` avant chaque appel. Au maximum 1536 nouveaux tokens. On sépare le raisonnement de la réponse au token 151668 (`</think>`).

## Résultats

### Évaluation de chaque prompt

Les prompts et les sorties complets sont dans [results/generations.md](results/generations.md). La pertinence veut dire que la réponse traite bien la question et que son contenu est correct. La cohérence veut dire que le texte est logique et bien construit. Le respect des consignes couvre les contraintes explicites (longueur, format, langue).

| Prompt | Check automatique | Pertinence | Cohérence | Respect des consignes | Commentaire |
|---|---|---|---|---|---|
| Instruction simple : définir un modèle de langage en deux phrases | 1 phrase, échec | ok | ok | échec | Définition correcte, mais une seule longue phrase au lieu de deux. |
| JSON avec les clés title, year, director, genres pour "Inception" | JSON valide, clés correctes, ok | ok | ok | ok | Se parse tel quel, sans bloc de code Markdown ; l'année 2010 et Christopher Nolan sont corrects. |
| Exactement 3 puces commençant par "- " | 0 puce valide, échec | partiel | partiel | partiel | Trois lignes, mais chacune commence par "- - ". Les trois avantages se recoupent, le deuxième décrit la fréquence des termes (les comptages bruts l'ont aussi) et le troisième (données en grande dimension) n'est pas un avantage par rapport aux comptages. |
| Paris en 15 mots maximum | 13 mots, ok | ok | ok | ok | Court, générique et correct. |
| Répondre en français (message system) | 7 stopwords français et 0 anglais, ok | ok | ok | ok | Entièrement en français. "Un bon avis d'oeuvre" est une formule bizarre pour une critique de film, le reste est correct. |
| Question factuelle, thinking off : réalisateur et année d'Amélie | pas de "Jeunet", pas de "2001", échec | échec | ok | ok | Répond "Jean-Luc Godard, 1961" avec assurance. La bonne réponse est Jean-Pierre Jeunet, 2001. |
| Même question, thinking on | pas de "Jeunet", pas de "2001", échec | échec | partiel | ok | Le raisonnement hésite entre Godard, un "François Zagoskin" inventé et "Jean-Pierre Jouet", puis revient à Godard, avec 1968 cette fois. |
| Problème d'arithmétique, thinking off (18 attendu) | nombre final 12, échec | échec | ok | partiel | Calcule bien les 12 livres de départ d'Anna, mais oublie le transfert de 6 livres. Finit par "Answer: <12>", avec les chevrons recopiés du template. |
| Même problème, thinking on | nombre final 18, ok | ok | ok | partiel | Raisonnement et réponse corrects, encore écrit "Answer: <18>". |
| Résumer un paragraphe en une phrase de 25 mots maximum | 1 phrase, 20 mots, ok | partiel | ok | ok | Respecte la longueur mais rate l'idée du paragraphe : le compromis entre l'entraînement spécifique à chaque tâche d'un côté, et la latence, la variabilité et les hallucinations de l'autre. |
| Intrigue d'un film qui n'existe pas | pas d'aveu d'ignorance, échec | échec | ok | ok | Invente une intrigue pour "Les Ombres de Valbrune" et ne dit jamais que le film est inconnu. |
| Sentiment de 8 critiques en français, zero-shot | 5/8 correctes | partiel | n/a | ok | Chaque réponse est un seul label, mais 7 sur 8 sont "positive". |
| Mêmes critiques, few-shot (4 exemples) | 6/8 correctes | partiel | n/a | ok | Corrige deux critiques négatives, casse une critique positive (une litote). |

### Classification de sentiment, zero-shot vs few-shot

C'est la tâche des TP 1 et 2 (sentiment de critiques de films en français), mais sans entraînement. Les 8 critiques sont des phrases courtes écrites pour ce test dans le style d'Allocine, elles ne viennent pas du dataset. On a mis les labels à la main, avec quelques cas difficiles (une litote, du sarcasme, des contrastes "X mais Y"). La consigne demande "positive" ou "negative", et la version few-shot ajoute 4 autres critiques du même genre comme tours user/assistant précédents.

| Critique | Attendu | Zero-shot | Few-shot |
|---|---|---|---|
| Un chef-d'œuvre absolu, des acteurs bouleversants. | positive | positive | positive |
| Deux heures d'ennui profond, je me suis endormi avant la fin. | negative | positive | negative |
| Je ne m'attendais pas à grand-chose, et pourtant j'ai adoré. | positive | positive | positive |
| Le casting était prometteur, mais le scénario est affligeant. | negative | positive | negative |
| Pas mal du tout, on ne voit pas le temps passer. | positive | positive | negative |
| Bravo, réussir à rendre ennuyeuse une histoire pareille, c'est un exploit. | negative | positive | positive |
| Des effets spéciaux à couper le souffle et une musique superbe. | positive | positive | positive |
| Rien à sauver, ni l'histoire, ni les dialogues, ni les acteurs. | negative | negative | negative |

En zero-shot, le modèle répond "positive" 7 fois sur 8 et ne repère que "Rien à sauver". Les exemples corrigent deux critiques négatives ("Deux heures d'ennui", et "Le casting était prometteur, mais...", qui a la même structure que le dernier exemple). Par contre, avec les exemples le modèle lit "Pas mal du tout" (une litote, donc positive) comme négative, probablement à cause de "pas" et "mal". Le "Bravo, ... c'est un exploit" sarcastique est mal classé dans les deux cas. Huit critiques, c'est beaucoup trop peu pour comparer sérieusement des accuracies. Le tableau montre surtout quels types de critiques sont difficiles.

### Vitesse sur CPU

Les temps par prompt sont dans le premier tableau de [results/generations.md](results/generations.md). Le temps inclut la lecture du prompt, donc les tokens/s ne donnent une vitesse de génération que pour les réponses longues : entre 5.2 et 8.2 tokens/s pour les réponses de 38 tokens ou plus. Les réponses courtes à des prompts longs ont l'air lentes. Le prompt de résumé contient tout le paragraphe et renvoie 24 tokens (2.5 tokens/s), et chaque appel de sentiment produit 2 tokens (le label et le token de fin de tour). Les appels few-shot prennent 50.1 s contre 18.5 s en zero-shot, pour les mêmes 16 tokens de sortie : on repaie les 4 exemples à chaque appel. Le lancement complet de `main.py` prend 405 s. Sur ce portable, les temps changent pas mal d'un lancement à l'autre, les sorties non.

### Thinking vs non-thinking

| Question | Mode | Tokens de thinking | Tokens au total | Temps (s) | Résultat |
|---|---|---:|---:|---:|---|
| Arithmétique | off | 0 | 108 | 13.2 | 12, faux |
| Arithmétique | on | 535 | 692 | 129.7 | 18, correct |
| Amélie | off | 0 | 44 | 7.5 | Godard, 1961, faux |
| Amélie | on | 523 | 568 | 90.1 | Godard, 1968, faux |

Le thinking corrige le problème d'arithmétique : la trace pose l'équation, trouve 12, puis relit la suite de la question et ajoute les 6 livres, en vérifiant le résultat plusieurs fois. Sans thinking, le modèle s'arrête dès qu'il a résolu l'équation. Sur la question factuelle, le thinking n'aide pas : la connaissance n'est pas dans le modèle, et la trace montre juste le modèle qui hésite entre plusieurs noms (l'un d'eux, "Jean-Pierre Jouet", est proche du bon) avant de revenir à sa première idée. Le coût est de 10 à 12 fois le temps de génération. Les deux lancements avec thinking utilisent aussi du sampling au lieu du greedy, donc la comparaison change deux choses en même temps.

### Limites observées

L'hallucination est le problème le plus grave : le mauvais réalisateur et la mauvaise année d'Amélie, ainsi que l'intrigue inventée, sont écrits sur le même ton sûr que les bonnes réponses, et le thinking n'y change rien. La dérive de format apparaît sur les contraintes moins courantes (marqueur de puce, placeholder de réponse, nombre de phrases) mais pas sur le JSON, qui est suivi exactement. Il n'y a pas de dérive de langue dans ces lancements : le message system en français est respecté, et le classifieur garde les labels anglais qu'on lui a donnés pour des critiques en français. Les réponses restent fluides même quand le contenu est faible, donc un check de format seul est trompeur, comme le montre le prompt de résumé.
