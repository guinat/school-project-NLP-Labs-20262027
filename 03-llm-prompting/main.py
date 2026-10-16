import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
# plus rapide que float32 au test de vitesse du début, mais fixé ici plutôt que choisi
# d'après ce test, pour que les sorties ne dépendent pas du bruit sur les temps
DTYPE = torch.bfloat16
# 2 cœurs performance et 8 cœurs efficacité : avec les 12 threads c'était bien plus lent
NUM_THREADS = 6
MAX_NEW_TOKENS = 256
MAX_THINKING_TOKENS = 1536
LABEL_MAX_NEW_TOKENS = 8
SPEED_TEST_TOKENS = 64
THINK_END_ID = 151668
SEED = 1337
GREEDY = {"do_sample": False}
# la model card prévient que le décodage greedy fait boucler le mode thinking
THINKING_SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
RESULTS_PATH = Path(__file__).parent / "results" / "generations.md"

SENTIMENT_INSTRUCTION = (
    "Classify the sentiment of the French movie review. "
    "Answer with a single word: positive or negative."
)

FEW_SHOT_EXAMPLES = [
    ("Un film captivant du début à la fin.", "positive"),
    ("Un film ennuyeux, j'ai regardé ma montre toutes les dix minutes.", "negative"),
    ("Pas un chef-d'œuvre, mais une comédie efficace qui fait rire.", "positive"),
    ("Les images sont belles, dommage que l'histoire soit aussi creuse.", "negative"),
]

REVIEWS = [
    ("Un chef-d'œuvre absolu, des acteurs bouleversants.", "positive"),
    ("Deux heures d'ennui profond, je me suis endormi avant la fin.", "negative"),
    ("Je ne m'attendais pas à grand-chose, et pourtant j'ai adoré.", "positive"),
    ("Le casting était prometteur, mais le scénario est affligeant.", "negative"),
    ("Pas mal du tout, on ne voit pas le temps passer.", "positive"),
    (
        "Bravo, réussir à rendre ennuyeuse une histoire pareille, c'est un exploit.",
        "negative",
    ),
    ("Des effets spéciaux à couper le souffle et une musique superbe.", "positive"),
    ("Rien à sauver, ni l'histoire, ni les dialogues, ni les acteurs.", "negative"),
]

PARAGRAPH = (
    "Sentiment analysis aims to decide whether a text expresses a positive or a "
    "negative opinion. A classical pipeline first cleans and tokenizes the reviews, "
    "then turns each one into a vector, for example with TF-IDF weights, and finally "
    "trains a linear classifier such as logistic regression on labeled examples. "
    "These models are fast, cheap and easy to inspect, but they need a new labeled "
    "dataset for every new task. Large language models take a different route: the "
    "task is described in the prompt, sometimes with a few examples, and the model "
    "answers without any task-specific training. This flexibility comes at the cost "
    "of higher latency, less predictable outputs and possible hallucinations."
)

AMELIE_QUESTION = (
    "Who directed the film Amélie (Le Fabuleux Destin d'Amélie Poulain) "
    "and in which year was it released?"
)

BOOKS_PROBLEM = (
    "Tom has three times as many books as Anna. Together they have 48 books. "
    "Tom then gives 6 books to Anna. How many books does Anna have now? "
    "End your answer with a line of the form 'Answer: <number>'."
)

JSON_KEYS = {"title", "year", "director", "genres"}
FRENCH_WORDS = {"le", "la", "les", "des", "un", "une", "est", "et", "du", "pour", "qui"}
ENGLISH_WORDS = {"the", "and", "is", "are", "of", "to", "in", "that", "with", "for"}
UNKNOWN_MARKERS = [
    "not aware",
    "no information",
    "not familiar",
    "don't have",
    "do not have",
    "could not find",
    "couldn't find",
    "does not exist",
    "doesn't exist",
    "no record",
    "unable to",
]


def chat(user, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": user}]


def words(text):
    return re.findall(r"[\w'-]+", text.lower())


def count_sentences(text):
    return len(re.findall(r"[.!?]+(?=\s|$)", text))


def check_sentences(text, expected):
    n = count_sentences(text)
    return {"sentences": n, "ok": n == expected}


def check_json(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"valid json": False, "ok": False}
    keys_ok = isinstance(data, dict) and set(data) == JSON_KEYS
    return {"valid json": True, "keys match": keys_ok, "ok": keys_ok}


def check_bullets(text, expected):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = sum(re.match(r"- \w", line) is not None for line in lines)
    other = len(lines) - bullets
    return {
        "bullets": bullets,
        "other lines": other,
        "ok": bullets == len(lines) == expected,
    }


def check_word_limit(text, limit):
    n = len(words(text))
    return {"words": n, "ok": n <= limit}


def check_french(text):
    tokens = words(text)
    french = sum(t in FRENCH_WORDS for t in tokens)
    english = sum(t in ENGLISH_WORDS for t in tokens)
    return {
        "french stopwords": french,
        "english stopwords": english,
        "ok": french > english,
    }


def check_contains(text, terms):
    found = {f"has {term}": term.lower() in text.lower() for term in terms}
    return {**found, "ok": all(found.values())}


def check_final_number(text, expected):
    numbers = re.findall(r"-?\d+", text)
    final = int(numbers[-1]) if numbers else None
    return {"final number": final, "ok": final == expected}


def check_summary(text, max_words):
    n_sentences = count_sentences(text)
    n_words = len(words(text))
    ok = n_sentences == 1 and n_words <= max_words
    return {"sentences": n_sentences, "words": n_words, "ok": ok}


def check_unknown(text):
    admits = any(marker in text.lower() for marker in UNKNOWN_MARKERS)
    return {"admits not knowing": admits, "ok": admits}


PROMPTS = [
    {
        "name": "simple instruction",
        "messages": chat("Explain in two sentences what a language model is."),
        "check": lambda text: check_sentences(text, 2),
    },
    {
        "name": "json format",
        "messages": chat(
            'Return a JSON object describing the movie "Inception" with exactly '
            'these keys: "title", "year", "director", "genres" (a list of strings). '
            "Output only the JSON, with no other text."
        ),
        "check": check_json,
    },
    {
        "name": "exactly 3 bullet points",
        "messages": chat(
            "Give exactly 3 advantages of TF-IDF over raw word counts for text "
            "classification. Write one bullet point per advantage, each line "
            'starting with "- ", and nothing else.'
        ),
        "check": lambda text: check_bullets(text, 3),
    },
    {
        "name": "word limit",
        "messages": chat("Describe the city of Paris in at most 15 words."),
        "check": lambda text: check_word_limit(text, 15),
    },
    {
        "name": "answer in French",
        "messages": chat(
            "What makes a good movie review?",
            system="You are a movie assistant. Always answer in French, "
            "whatever the language of the question.",
        ),
        "check": check_french,
    },
    {
        "name": "factual question, thinking off",
        "messages": chat(AMELIE_QUESTION),
        "check": lambda text: check_contains(text, ["Jeunet", "2001"]),
    },
    {
        "name": "factual question, thinking on",
        "messages": chat(AMELIE_QUESTION),
        "check": lambda text: check_contains(text, ["Jeunet", "2001"]),
        "thinking": True,
    },
    {
        "name": "arithmetic, thinking off",
        "messages": chat(BOOKS_PROBLEM),
        "check": lambda text: check_final_number(text, 18),
    },
    {
        "name": "arithmetic, thinking on",
        "messages": chat(BOOKS_PROBLEM),
        "check": lambda text: check_final_number(text, 18),
        "thinking": True,
    },
    {
        "name": "summarization",
        "messages": chat(
            "Summarize the following paragraph in one sentence of at most 25 words."
            f'\n\n"""\n{PARAGRAPH}\n"""'
        ),
        "check": lambda text: check_summary(text, 25),
    },
    {
        "name": "fictitious entity",
        "messages": chat(
            'Summarize the plot of the 1974 French film "Les Ombres de Valbrune" '
            "directed by Étienne Vardoux."
        ),
        "check": check_unknown,
    },
]


def generate(model, tokenizer, messages, thinking=False, max_new_tokens=MAX_NEW_TOKENS):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    inputs = tokenizer(text, return_tensors="pt")
    decoding = THINKING_SAMPLING if thinking else GREEDY
    torch.manual_seed(SEED)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, **decoding)
    seconds = time.perf_counter() - start

    new_ids = output[0, inputs.input_ids.shape[1] :].tolist()
    split = new_ids.index(THINK_END_ID) + 1 if THINK_END_ID in new_ids else 0
    reasoning = tokenizer.decode(new_ids[:split], skip_special_tokens=True)
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
    answer = tokenizer.decode(new_ids[split:], skip_special_tokens=True).strip()
    return {
        "answer": answer,
        "reasoning": reasoning,
        "reasoning_tokens": split,
        "new_tokens": len(new_ids),
        "seconds": seconds,
    }


def speed_test(model, tokenizer):
    messages = chat("Write a short story about a cinema.")
    # warm-up : le premier appel est plus lent
    generate(model, tokenizer, messages, max_new_tokens=8)
    run = generate(model, tokenizer, messages, max_new_tokens=SPEED_TEST_TOKENS)
    speed = run["new_tokens"] / run["seconds"]
    print(f"{model.dtype}: {speed:.2f} tokens/s")
    return speed


def classify_reviews(model, tokenizer, few_shot):
    messages = [{"role": "system", "content": SENTIMENT_INSTRUCTION}]
    if few_shot:
        for review, label in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": review})
            messages.append({"role": "assistant", "content": label})
    runs = []
    for review, expected in REVIEWS:
        prompt = messages + [{"role": "user", "content": review}]
        run = generate(model, tokenizer, prompt, max_new_tokens=LABEL_MAX_NEW_TOKENS)
        run["correct"] = run["answer"].lower().strip(" .") == expected
        runs.append(run)
    return runs


def format_checks(checks):
    return ", ".join(f"{key}: {value}" for key, value in checks.items())


def table_row(name, runs, checks):
    tokens = sum(run["new_tokens"] for run in runs)
    seconds = sum(run["seconds"] for run in runs)
    return f"| {name} | {tokens} | {seconds:.1f} | {tokens / seconds:.1f} | {checks} |"


def write_report(runs, sentiment, speeds):
    speed_line = ", ".join(f"{name} {rate:.2f}" for name, rate in speeds.items())
    lines = [
        "# Generations",
        "",
        f"Model: {MODEL_NAME}, {DTYPE}, CPU with {NUM_THREADS} threads.",
        f"Speed test ({SPEED_TEST_TOKENS} greedy tokens, tokens/s): {speed_line}.",
        "Thinking off: greedy decoding. Thinking on: sampling with temperature 0.6, "
        f"top-p 0.95, top-k 20 and seed {SEED}.",
        "",
        "| prompt | new tokens | time (s) | tokens/s | checks |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        lines.append(table_row(run["name"], [run], format_checks(run["checks"])))
    for name, sentiment_runs in sentiment.items():
        correct = sum(run["correct"] for run in sentiment_runs)
        accuracy = f"accuracy: {correct}/{len(sentiment_runs)}"
        calls = f"sentiment, {name} ({len(sentiment_runs)} calls)"
        lines.append(table_row(calls, sentiment_runs, accuracy))

    for run in runs:
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in run["messages"])
        lines += [
            "",
            f"## {run['name']}",
            "",
            f"Checks: {format_checks(run['checks'])}",
        ]
        lines += ["", "Prompt:", "", "````text", prompt, "````", ""]
        if run["reasoning_tokens"]:
            lines += [f"Thinking ({run['reasoning_tokens']} tokens):", ""]
            lines += ["````text", run["reasoning"], "````", ""]
        lines += ["Output:", "", "````text", run["answer"], "````"]

    lines += ["", "## sentiment classification", "", f"System: {SENTIMENT_INSTRUCTION}"]
    lines += ["", "Few-shot examples, passed as previous user/assistant turns:", ""]
    lines += [f"- {review} -> {label}" for review, label in FEW_SHOT_EXAMPLES]
    lines += ["", "| review | expected | zero-shot | few-shot |", "|---|---|---|---|"]
    for (review, expected), zero, few in zip(
        REVIEWS, sentiment["zero-shot"], sentiment["few-shot"], strict=True
    ):
        lines.append(f"| {review} | {expected} | {zero['answer']} | {few['answer']} |")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


def main():
    start = time.perf_counter()
    torch.set_num_threads(NUM_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    fp32_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    speeds = {torch.float32: speed_test(fp32_model, tokenizer)}
    del fp32_model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    speeds[DTYPE] = speed_test(model, tokenizer)
    print(f"using {DTYPE}, {NUM_THREADS} threads")

    runs = []
    for prompt in PROMPTS:
        thinking = prompt.get("thinking", False)
        max_new_tokens = MAX_THINKING_TOKENS if thinking else MAX_NEW_TOKENS
        run = generate(model, tokenizer, prompt["messages"], thinking, max_new_tokens)
        run.update(name=prompt["name"], messages=prompt["messages"])
        run["checks"] = prompt["check"](run["answer"])
        runs.append(run)

        print(f"\n[{run['name']}]")
        if thinking:
            print(f"thinking ({run['reasoning_tokens']} tokens): {run['reasoning']}\n")
        print(run["answer"])
        rate = run["new_tokens"] / run["seconds"]
        print(
            f"{run['new_tokens']} tokens, {run['seconds']:.1f} s, {rate:.1f} tokens/s"
        )
        print(format_checks(run["checks"]))

    sentiment = {
        "zero-shot": classify_reviews(model, tokenizer, few_shot=False),
        "few-shot": classify_reviews(model, tokenizer, few_shot=True),
    }
    print("\nexpected | zero-shot | few-shot | review")
    for (review, expected), zero, few in zip(
        REVIEWS, sentiment["zero-shot"], sentiment["few-shot"], strict=True
    ):
        print(f"{expected} | {zero['answer']} | {few['answer']} | {review}")
    for name, sentiment_runs in sentiment.items():
        correct = sum(run["correct"] for run in sentiment_runs)
        print(f"{name} accuracy: {correct}/{len(sentiment_runs)}")

    write_report(runs, sentiment, speeds)
    print(f"\nwrote {RESULTS_PATH}")
    print(f"total time: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
