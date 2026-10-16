# NLP labs

TP des cours de NLP et de Transformers (ING3). Chaque dossier est un projet [uv](https://docs.astral.sh/uv/) à part, avec ses propres dépendances.

- `01-sentiment-classical` : NLP 1, analyse de sentiment avec Naive Bayes et régression logistique
- `02-sentiment-mlp` : NLP 2, MLP pour l'analyse de sentiment
- `03-llm-prompting` : NLP 3, prompting de Qwen3-0.6B
- `04-bigram-language-model` : Transformers 1, modèle de langage bigramme
- `05-self-attention` : Transformers 2, self-attention causale
- `06-layernorm-feedforward` : Transformers 3, LayerNorm, feed-forward et connexion résiduelle
- `07-gpt-from-scratch` : Transformers 4, petit GPT entraîné sur Tiny Shakespeare

## Lancer un TP

```bash
cd 04-bigram-language-model
uv run main.py
```

Il faut juste avoir uv installé (il installe Python 3.12 si besoin). Tout tourne sur CPU.

Les TP Transformers suivent la vidéo [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) de Karpathy.
