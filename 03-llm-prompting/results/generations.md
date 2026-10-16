# Generations

Model: Qwen/Qwen3-0.6B, torch.bfloat16, CPU with 6 threads.
Speed test (64 greedy tokens, tokens/s): torch.float32 6.19, torch.bfloat16 6.48.
Thinking off: greedy decoding. Thinking on: sampling with temperature 0.6, top-p 0.95, top-k 20 and seed 1337.

| prompt | new tokens | time (s) | tokens/s | checks |
|---|---|---|---|---|
| simple instruction | 38 | 6.1 | 6.2 | sentences: 1, ok: False |
| json format | 44 | 8.4 | 5.2 | valid json: True, keys match: True, ok: True |
| exactly 3 bullet points | 89 | 14.9 | 6.0 | bullets: 0, other lines: 3, ok: False |
| word limit | 17 | 3.5 | 4.8 | words: 13, ok: True |
| answer in French | 53 | 9.2 | 5.7 | french stopwords: 7, english stopwords: 0, ok: True |
| factual question, thinking off | 44 | 7.5 | 5.9 | has Jeunet: False, has 2001: False, ok: False |
| factual question, thinking on | 568 | 90.1 | 6.3 | has Jeunet: False, has 2001: False, ok: False |
| arithmetic, thinking off | 108 | 13.2 | 8.2 | final number: 12, ok: False |
| arithmetic, thinking on | 692 | 129.7 | 5.3 | final number: 18, ok: True |
| summarization | 24 | 9.7 | 2.5 | sentences: 1, words: 20, ok: True |
| fictitious entity | 108 | 17.4 | 6.2 | admits not knowing: False, ok: False |
| sentiment, zero-shot (8 calls) | 16 | 18.5 | 0.9 | accuracy: 5/8 |
| sentiment, few-shot (8 calls) | 16 | 50.1 | 0.3 | accuracy: 6/8 |

## simple instruction

Checks: sentences: 1, ok: False

Prompt:

````text
[user]
Explain in two sentences what a language model is.
````

Output:

````text
A language model is a type of artificial intelligence that can understand and generate human language, learning from vast amounts of text data to perform tasks like translation, text generation, and summarization.
````

## json format

Checks: valid json: True, keys match: True, ok: True

Prompt:

````text
[user]
Return a JSON object describing the movie "Inception" with exactly these keys: "title", "year", "director", "genres" (a list of strings). Output only the JSON, with no other text.
````

Output:

````text
{
  "title": "Inception",
  "year": 2010,
  "director": "Christopher Nolan",
  "genres": ["action", "sci-fi", "drama"]
}
````

## exactly 3 bullet points

Checks: bullets: 0, other lines: 3, ok: False

Prompt:

````text
[user]
Give exactly 3 advantages of TF-IDF over raw word counts for text classification. Write one bullet point per advantage, each line starting with "- ", and nothing else.
````

Output:

````text
- - TF-IDF improves text classification by incorporating both word frequency and word importance, which helps in identifying relevant terms.  
- - TF-IDF considers the frequency of a word in a document, which helps in capturing the importance of a word in the context of the document.  
- - TF-IDF is more effective in handling high-dimensional text data compared to raw word counts, as it accounts for both word frequency and word importance.
````

## word limit

Checks: words: 13, ok: True

Prompt:

````text
[user]
Describe the city of Paris in at most 15 words.
````

Output:

````text
Paris is a vibrant city known for its history, culture, and artistic heritage.
````

## answer in French

Checks: french stopwords: 7, english stopwords: 0, ok: True

Prompt:

````text
[system]
You are a movie assistant. Always answer in French, whatever the language of the question.

[user]
What makes a good movie review?
````

Output:

````text
Un bon avis d'oeuvre se base sur plusieurs facteurs : la qualité du contenu, l'originalité, la crédibilité, la capacité à capturer l'immersion du spectateur, et l'impact sur l'audience.
````

## factual question, thinking off

Checks: has Jeunet: False, has 2001: False, ok: False

Prompt:

````text
[user]
Who directed the film Amélie (Le Fabuleux Destin d'Amélie Poulain) and in which year was it released?
````

Output:

````text
The film *Amélie* (Le Fabuleux Destin d'Amélie Poulain) was directed by **Jean-Luc Godard** and released in **1961**.
````

## factual question, thinking on

Checks: has Jeunet: False, has 2001: False, ok: False

Prompt:

````text
[user]
Who directed the film Amélie (Le Fabuleux Destin d'Amélie Poulain) and in which year was it released?
````

Thinking (523 tokens):

````text
Okay, so I need to figure out who directed the film "Amélie" and the year it was released. Let me start by recalling what I know about this film. I remember that Amélie is a French film, directed by someone, and it's about a girl named Amélie who goes through various challenges. 

First, I think the director is Jean-Luc Godard. Wait, but I'm not sure. I've heard that Godard directed films like "La Vie en Rose" and "La Belle Dame de Paris," which are part of his style. But is "Amélie" another one of his works? I think I might be mixing up some details here. 

Wait, maybe it's not Jean-Luc Godard. Let me think again. Another possibility is that the director is someone else. I recall that "Amélie" was directed by someone named François Zagoskin, but I'm not sure. Alternatively, maybe it's directed by a different person. 

Alternatively, I might be confusing it with "Amélie" from the 1990s. Wait, no, that's not right. Let me check my memory. Oh, right! I think the correct director is Jean-Luc Godard, but maybe I'm mixing up with another film. Wait, no, maybe it's not Jean-Luc Godard. Let me think of the original French film. 

Wait, I think I need to verify this. The original French film "Amélie" was directed by Jean-Luc Godard, released in 1968. But I'm not entirely sure. Alternatively, maybe it's a different film. Let me try to recall. 

Alternatively, maybe it's directed by someone else. For example, "Amélie" could be directed by someone like Jean-Pierre Jouet. But I'm not sure. 

Wait, I think I'm mixing up the film with another. Let me think again. The original French film "Amélie" was directed by Jean-Luc Godard in 1968. So the answer would be that the director is Jean-Luc Godard, and the year is 1968. 

But I need to make sure. Let me try to think of the plot. "Amélie" is a story about a girl who goes through various trials, which is similar to other films. If I remember correctly, Jean-Luc Godard directed this film. So I think the answer is correct.
````

Output:

````text
The film *Amélie* (Le Fabuleux Destin d'Amélie Poulain) was directed by **Jean-Luc Godard** and released in **1968**.
````

## arithmetic, thinking off

Checks: final number: 12, ok: False

Prompt:

````text
[user]
Tom has three times as many books as Anna. Together they have 48 books. Tom then gives 6 books to Anna. How many books does Anna have now? End your answer with a line of the form 'Answer: <number>'.
````

Output:

````text
Let’s define the number of books Anna has as $ x $. Then:

- Tom has $ 3x $ books.
- Together, they have $ 3x + x = 4x $ books.
- They have 48 books in total, so:
  $$
  4x = 48
  $$
  Solving for $ x $:
  $$
  x = 12
  $$

So, Anna has **12 books**.

Answer: <12>
````

## arithmetic, thinking on

Checks: final number: 18, ok: True

Prompt:

````text
[user]
Tom has three times as many books as Anna. Together they have 48 books. Tom then gives 6 books to Anna. How many books does Anna have now? End your answer with a line of the form 'Answer: <number>'.
````

Thinking (535 tokens):

````text
Okay, let's see. So, the problem says that Tom has three times as many books as Anna. Together they have 48 books. Then Tom gives 6 books to Anna. I need to find out how many books Anna has now. Hmm, let me break this down step by step.

First, let me assign variables to the number of books each person has. Let me let Anna's books be a. Then, since Tom has three times as many as Anna, Tom's books would be 3a. Together, they have 48 books, so adding both together: a + 3a = 48. Let me check that. If Anna has a and Tom has 3a, then total is indeed a + 3a = 4a. Wait, so 4a equals 48? That would mean a equals 12. So Anna starts with 12 books, and Tom has 3 times that, so 36 books. Let me confirm that. 12 + 36 is 48, which matches. Okay, so that seems right.

Now, Tom gives 6 books to Anna. So Anna's new number of books would be her original 12 plus 6, which is 18. So Anna now has 18 books. Let me make sure I didn't make any mistakes here. Let me redo the math. If total is 48, and Tom has 3a, Anna has a. So total is 3a + a = 4a = 48. Therefore, a = 12. Then Tom gives 6 books, so Anna gets 6, so 12 + 6 = 18. Yeah, that seems correct.

Wait, but maybe I need to check if Tom gives 6 books to Anna, so Anna's books go up by 6, and Tom's go down by 6. So Anna's books after the transfer would be 12 + 6 = 18. That's straightforward. So the answer should be 18. Let me write that in the required format. Answer: <number>. So, 18. But let me check once more to be sure.

Original numbers: Anna has 12, Tom has 36. Total 48. Then Tom gives 6 to Anna. So Anna now has 12 + 6 = 18. Yes, that's correct. I don't think I made any mistakes here. So the answer is 18.
````

Output:

````text
Tom has three times as many books as Anna, and together they have 48 books. Let's denote Anna's initial number of books as $ a $. Then Tom's number of books is $ 3a $. Together, their total is:

$$
a + 3a = 4a = 48
$$

Solving for $ a $:

$$
a = \frac{48}{4} = 12
$$

So Anna initially has 12 books, and Tom has $ 3 \times 12 = 36 $ books. After Tom gives 6 books to Anna, Anna's total becomes:

$$
12 + 6 = 18
$$

Answer: <18>
````

## summarization

Checks: sentences: 1, words: 20, ok: True

Prompt:

````text
[user]
Summarize the following paragraph in one sentence of at most 25 words.

"""
Sentiment analysis aims to decide whether a text expresses a positive or a negative opinion. A classical pipeline first cleans and tokenizes the reviews, then turns each one into a vector, for example with TF-IDF weights, and finally trains a linear classifier such as logistic regression on labeled examples. These models are fast, cheap and easy to inspect, but they need a new labeled dataset for every new task. Large language models take a different route: the task is described in the prompt, sometimes with a few examples, and the model answers without any task-specific training. This flexibility comes at the cost of higher latency, less predictable outputs and possible hallucinations.
"""
````

Output:

````text
Sentiment analysis uses a pipeline to analyze text, with linear models or large language models differing in approach and performance.
````

## fictitious entity

Checks: admits not knowing: False, ok: False

Prompt:

````text
[user]
Summarize the plot of the 1974 French film "Les Ombres de Valbrune" directed by Étienne Vardoux.
````

Output:

````text
**"Les Ombres de Valbrune" (1974)**, directed by Étienne Vardoux, is a French film that follows the life of Valbrune, a young man who becomes a shadowy figure in the world. The story is set in a small French town and follows Valbrune's journey as he navigates the complexities of his life, relationships, and personal struggles. The film explores themes of identity, love, and the search for meaning in a world that often feels indifferent.
````

## sentiment classification

System: Classify the sentiment of the French movie review. Answer with a single word: positive or negative.

Few-shot examples, passed as previous user/assistant turns:

- Un film captivant du début à la fin. -> positive
- Un film ennuyeux, j'ai regardé ma montre toutes les dix minutes. -> negative
- Pas un chef-d'œuvre, mais une comédie efficace qui fait rire. -> positive
- Les images sont belles, dommage que l'histoire soit aussi creuse. -> negative

| review | expected | zero-shot | few-shot |
|---|---|---|---|
| Un chef-d'œuvre absolu, des acteurs bouleversants. | positive | positive | positive |
| Deux heures d'ennui profond, je me suis endormi avant la fin. | negative | positive | negative |
| Je ne m'attendais pas à grand-chose, et pourtant j'ai adoré. | positive | positive | positive |
| Le casting était prometteur, mais le scénario est affligeant. | negative | positive | negative |
| Pas mal du tout, on ne voit pas le temps passer. | positive | positive | negative |
| Bravo, réussir à rendre ennuyeuse une histoire pareille, c'est un exploit. | negative | positive | positive |
| Des effets spéciaux à couper le souffle et une musique superbe. | positive | positive | positive |
| Rien à sauver, ni l'histoire, ni les dialogues, ni les acteurs. | negative | negative | negative |
