import pandas as pd
import numpy as np
import os
import yaml
import pickle
from analyze import (
    experiment_set,
    NarrativeContext,
    join_many_strings,
    get_logits,
    likelihood_delta,
    stratified_n_tuple_sampling,
)
from tqdm.auto import tqdm
import random
import os
from import_data import import_data
import pandas as pd
import json
import yaml

# Model
MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

# Conditioners
CONDITIONERS = 2

# Sample rate parameters
SAMPLE_SIZE = -1
SAMPLE_COUNT = -1
SAMPLE_MULTIPLIER = 1

# Corpus
# CORPUS_NAME = "climate_change"
CORPUS_NAME = "dog_ownership"
# CORPUS_NAME = "domestic_violence"
# CORPUS_NAME = "misogyny"

# The name of the experiment (i.e. where to save the results)
EXPERIMENT_NAME = "script-test-experiment"

# Whether we're in debug mode
DEBUG = False

# Corpus Size
CORPUS_SIZE = "/20_10"

# The dataset type
DATASET_TYPE = "distilled"
# DATASET_TYPE = "summaries"

CORPUS = "../gpt-4-only-corpora/" + CORPUS_NAME + CORPUS_SIZE + ".json"

# Replace @ and / in the experiment name
EXPERIMENT_NAME = EXPERIMENT_NAME.replace("@", "_").replace("/", "_")

ctx = NarrativeContext(
    model=MODEL,
    conditioners=CONDITIONERS,
    # sample_rate=SAMPLE_SIZE
)

# Create the results directory
RESULTS_DIR = os.path.join("corpus_results/diff_classification", EXPERIMENT_NAME)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def get_seeds(path):
    with open(path, "r") as f:
        corpus_data = yaml.safe_load(f)

    seeds = corpus_data["seeds"]
    distilled = corpus_data["distilled"]
    summarized = corpus_data["summarized"]
    names = corpus_data["names"]
    dataset = corpus_data["dataset"]

    if DATASET_TYPE == "distilled":
        a_s = []
        b_s = []
        for seed_pair in distilled:
            a_s.append(seed_pair["a"])
            b_s.append(seed_pair["b"])
    elif DATASET_TYPE == "summaries":
        a_s = [summarized["a"]]
        b_s = [summarized["b"]]
    else:
        raise ValueError("Invalid dataset type: {}".format(DATASET_TYPE))

    print("a", len(a_s))
    print("b", len(b_s))
    return a_s, b_s, []


def get_sentences(path):
    with open(path, "r") as f:
        corpus_data = yaml.safe_load(f)

    seeds = corpus_data["seeds"]
    distilled = corpus_data["distilled"]
    summarized = corpus_data["summarized"]
    names = corpus_data["names"]
    dataset = corpus_data["dataset"]

    a_s = []
    b_s = []
    for datum in dataset:
        a_first = datum["a_first"]
        b_first = datum["b_first"]

        # Add a sentences
        a_s += a_first["a"]
        a_s += b_first["a"]

        # Add b sentences
        b_s += a_first["b"]
        b_s += b_first["b"]

    print("a", len(a_s))
    print("b", len(b_s))

    # Shuffle a_s and b_s
    random.shuffle(a_s)
    random.shuffle(b_s)

    # Return
    return a_s, b_s, []


def align_text_to_conditioners(label, narrative_sets, text, ctx, sample_size):
    # No batching
    deltas = []
    for conditioners in narrative_sets:
        conditioner_string = join_many_strings(conditioners)
        deltas.append(
            {
                "conditioners": conditioners,
                "likelihood_delta": likelihood_delta(
                    conditioner_string, text["text"], ctx
                ),
            }
        )

    return {"text": text, "likelihood_deltas": deltas}


def align(label, narrative_set, texts, sample_size):
    print("*** Aligning {} texts to {} ***".format(len(texts), label))
    results = []
    for text in tqdm(texts):
        results.append(
            align_text_to_conditioners(
                label,
                pairs[(tuple(narrative_set), text["text"])],
                text,
                ctx,
                sample_size,
            )
        )

    return results


# Load texts from corpora/{CORPUS}.json
if CORPUS_NAME in ("voice", "climate", "asylum_seekers"):
    X, y, seeds, distilled, summarized, names = import_data(CORPUS)

    helpful = [seed["a"] for seed in seeds]
    unhelpful = [seed["b"] for seed in seeds]
    neutral = []

    a_sentences = []
    b_sentences = []
    for _x, _y in zip(X, y):
        if _y == "a":
            a_sentences.append(_x)
        elif _y == "b":
            b_sentences.append(_x)
        else:
            raise ValueError("Invalid label: {}".format(_y))

    sentences = a_sentences + b_sentences
    random.shuffle(sentences)
    corpus_data = [
        {"text": text, "speakername": label, "date": "2024-03-06"}
        for text, label in zip(sentences, y)
    ]

else:
    helpful, unhelpful, neutral = get_seeds(CORPUS)
    a_sentences, b_sentences, _ = get_sentences(CORPUS)
    # sentences = a_sentences + b_sentences
    sentences = []
    for a in a_sentences:
        sentences.append((a, "a"))
    for b in b_sentences:
        sentences.append((b, "b"))
    random.shuffle(sentences)
    corpus_data = [
        {"text": text, "speakername": label, "date": "2024-03-06"}
        for text, label in sentences
    ]

# # Debug, get only 10 sentences
# corpus_data = corpus_data[:10]

print("Loaded {} texts from corpus".format(len(corpus_data)))
print("Total word count:", sum([len(text["text"].split()) for text in corpus_data]))

# Use SAMPLE_COUNT
if SAMPLE_COUNT != -1 and len(corpus_data) > SAMPLE_COUNT:
    corpus_data = random.sample(corpus_data, SAMPLE_COUNT)


# Precompute narrative set -- text pairs
pairs = {}
to_precompute = set()
for narrative_set in [helpful, unhelpful, neutral]:
    for text in corpus_data:
        # # Pick a random narrative set
        # narrative_sets = list(itertools.permutations(narrative_set, CONDITIONERS))
        # narrative_sets = random.sample(narrative_sets, min(len(narrative_sets), len(narrative_set)))
        narrative_sets = []
        for _ in range(SAMPLE_MULTIPLIER):
            narrative_sets_iter = stratified_n_tuple_sampling(
                list(narrative_set), CONDITIONERS
            )
            narrative_sets.extend(narrative_sets_iter)

        # Append to the list
        pairs[(tuple(narrative_set), text["text"])] = narrative_sets

        # Add to the set of things to precompute
        for conditioners in narrative_sets:
            conditioner_string = join_many_strings(conditioners)
            to_precompute.add(join_many_strings([conditioner_string, text["text"]]))
            to_precompute.add(conditioner_string)
            to_precompute.add(text["text"])

# Get the total word count of the precompute set
print(
    "Total word count of precompute set:",
    sum([len(text.split()) for text in to_precompute]),
)
# break

# Precompute the logits
print("--- Precomputing logits ---")
logits, usages = get_logits(list(to_precompute), ctx.model)
print(usages[0])
print("--- Done precomputing logits ---")

# Run experiments
results = {
    "helpful": align("helpful", helpful, corpus_data, None),
    "unhelpful": align("unhelpful", unhelpful, corpus_data, None),
    # "neutral": align("neutral", neutral, corpus_data, SAMPLE_SIZE)
}

# Save results with pickle
with open(
    os.path.join(
        "corpus_results/diff_classification",
        EXPERIMENT_NAME,
        # f"{CORPUS}_{MODEL}_{CONDITIONERS}_{SAMPLE_MULTIPLIER}.pkl",
        "results.pkl",
    ),
    "wb",
) as f:
    pickle.dump(results, f)

# Save usage data to CSV
# Usage object is like
# CompletionUsage(completion_tokens=1, prompt_tokens=43, total_tokens=44)
# We want to save this as two columns - prompt_tokens and completion_tokens

usage_data = []
for usage in usages:
    usage_data.append(
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
    )

usage_df = pd.DataFrame(usage_data)
usage_df.to_csv(
    os.path.join(
        "corpus_results/diff_classification",
        EXPERIMENT_NAME,
        # f"{CORPUS}_{MODEL}_{CONDITIONERS}_{SAMPLE_MULTIPLIER}_usage.csv",
        "usage.csv",
    ),
    index=False,
)

# Load data from pkl
with open(
    os.path.join(
        "corpus_results/diff_classification",
        EXPERIMENT_NAME,
        # f"{CORPUS}_{MODEL}_{CONDITIONERS}_{SAMPLE_MULTIPLIER}.pkl",
        "results.pkl",
    ),
    "rb",
) as f:
    results = pickle.load(f)

# For each narrative set in (helpful, unhelpful), find the texts with the highest and lowest likelihood delta (relative to the other set
# of narratives) and print them.
helpful = results["helpful"]
unhelpful = results["unhelpful"]


def compute_deltas(result_set):
    # Compute generally for helpful/unhelpful/neutral
    deltas = {}
    for text in result_set:
        text_deltas = [
            delta["likelihood_delta"]["likelihood_delta"]
            for delta in text["likelihood_deltas"]
        ]
        deltas[text["text"]["text"]] = np.mean(text_deltas)
        # Get the signed max abs value
        # deltas[text["text"]["text"]] = max(text_deltas, key=abs)
        # deltas[text["text"]["text"]] = max(text_deltas)
        print(text)
    return deltas


helpful_deltas = compute_deltas(helpful)
unhelpful_deltas = compute_deltas(unhelpful)

helpful_unhelpful_diffs = []
for text in helpful:
    helpful_unhelpful_diffs.append(
        {
            "text": text["text"]["text"],
            "helpful": helpful_deltas[text["text"]["text"]],
            "unhelpful": unhelpful_deltas[text["text"]["text"]],
            # "helpful_deltas": [delta["likelihood_delta"]["likelihood_delta"] for delta in text["likelihood_deltas"]],
            # "unhelpful_deltas": [delta["likelihood_delta"]["likelihood_delta"] for delta in text["likelihood_deltas"]],
            "speaker": text["text"]["speakername"],
            "affiliation": (
                text["text"]["affiliation"] if "affiliation" in text["text"] else None
            ),
            "date": text["text"]["date"],
            "diff": helpful_deltas[text["text"]["text"]]
            - unhelpful_deltas[text["text"]["text"]],
        }
    )

helpful_unhelpful_diffs = sorted(helpful_unhelpful_diffs, key=lambda x: x["diff"])

print("Most unhelpful:")
for text in helpful_unhelpful_diffs[:10]:
    print(f"[{text['diff']}] {text['text']}")
print()
print("Most helpful:")
for text in helpful_unhelpful_diffs[-10:]:
    print(f"[{text['diff']}] {text['text']}")
print()

outpath = os.path.join(
    "corpus_results/diff_classification",
    EXPERIMENT_NAME,
    # f"{CORPUS}_{MODEL}_{CONDITIONERS}_{SAMPLE_MULTIPLIER}",
    "results",
)
print("Saving to:", outpath)

# Save results with JSON
with open(outpath + ".json", "w") as f:
    json.dump(helpful_unhelpful_diffs, f, indent=4)

# Save in CSV format
df = pd.DataFrame(helpful_unhelpful_diffs)
df.to_csv(outpath + ".csv", index=False)
