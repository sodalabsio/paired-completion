import string
import scipy.stats as stats
import openai
from tqdm.auto import tqdm
import itertools
from dataclasses import dataclass
import random
from typing import List, Union, Tuple, Dict
from soda.openai.text import completion_model_batched


@dataclass
class NarrativeContext:
    model: str = "davinci"
    conditioners: int = 1
    sample_rate: float = 1.0
    num_threads: int = 16


db = {}


class VALUE_CACHE:
    # Cached values go from (model_name, text) -> value
    @staticmethod
    def get(model_name: str, text: str, allow_miss=True) -> Union[float, None]:
        # Get the value from the cache
        key = model_name + "||" + text
        if key in db:
            return db[key]

        if allow_miss:
            return None

        raise Exception("Cache miss on:", key)

    @staticmethod
    def get_batch(
        model_name: str, texts: List[str], allow_miss=True
    ) -> Union[List[float], None]:
        # Get the value from the cache
        keys = [model_name + "||" + text for text in texts]
        values = [db[key] if key in db else None for key in keys]

        if allow_miss:
            return values

        if None in values:
            raise Exception("Cache miss on:", keys)
        return values

    @staticmethod
    def set(model_name: str, text: str, value: float):
        # Set the value in the cache
        key = model_name + "||" + text
        db[key] = value


def get_logits(texts: List[str], model_name: str, num_threads: int = 16):
    to_cache = []

    # Get the values that are not in the cache
    from_cache = VALUE_CACHE.get_batch(model_name, texts, allow_miss=True)
    for text, value in zip(texts, from_cache):
        if value is None:
            to_cache.append(text)

    # Shuffle to_cache in case there are sequential duplicates
    random.shuffle(to_cache)

    # We always use the OpenAI API
    uncached_logits, uncached_usages = get_openai_logits(
        to_cache, model_name, num_threads=num_threads
    )

    # Cache the uncached values
    for text, (logit_sum, usage) in zip(
        to_cache, zip(uncached_logits, uncached_usages)
    ):
        # print("Setting value {} for text '{}'".format(logit_sum, text))
        VALUE_CACHE.set(model_name, text, (logit_sum, usage))

    # All values are now in the cache
    logits = [VALUE_CACHE.get(model_name, text, allow_miss=False)[0] for text in texts]
    usages = [VALUE_CACHE.get(model_name, text, allow_miss=False)[1] for text in texts]

    # return logits
    return logits, usages


def get_llama_logits(texts: List[str], model_name: str) -> List[float]:
    # Texts is a (potentially large) list of strings, for which we need
    # to find the logits of each token in each string
    # We then cache these values

    # Do the import here to avoid annoying waits when we're not even using llama
    from llama_wrapper import get_logprobs

    values = []
    for text in tqdm(texts):
        logprob_sum, _ = get_logprobs(texts, model_name)
        values.append(logprob_sum)

    # Get logits for each input token
    return values


def get_openai_logits(
    texts: List[str], model_name: str, batch_size: int = 20, num_threads: int = 20
):
    if len(texts) == 0:
        return []

    results = completion_model_batched(
        x=texts,
        model=model_name,
        batch_size=1,
        num_threads=num_threads,
        max_tokens=1,
        temperature=0,
        # Get token probabilities
        logprobs=1,
        # Get the logprobs of the actual tokens by getting the logit at the given index
        echo=True,
    )

    logits = []
    usages = []
    for result in results:
        log_probs = result.choices[0].logprobs.token_logprobs
        logits.append(sum(p for p in log_probs[:-1] if p is not None))
        usages.append(result.usage)

    return logits, usages


def p(x: str, ctx: NarrativeContext) -> float:
    # We pull from the cache (assuming the (text, model) pair is already there)
    cached_value, _ = VALUE_CACHE.get(ctx.model, x, allow_miss=False)
    assert cached_value is not None

    return cached_value


def join_strings(a: str, b: str) -> str:
    assert type(a) is str, f"Expected str, got {type(a)}"
    assert type(b) is str, f"Expected str, got {type(b)}"

    a = a.strip()
    b = b.strip()

    if len(a) == 0:
        return b

    if len(b) == 0:
        return a

    a_last = a[-1]
    if a_last in string.punctuation:
        return a + " " + b

    return a + ". " + b


# def join_strings(a: str, b: str) -> str:
#     return a + " " + b


def join_many_strings(s: List[str]) -> str:
    if type(s) is str:
        raise Exception("Expected list of strings, got string")

    assert type(s) in (list, tuple), f"Expected list, got '{s}' of type {type(s)}"
    whole = ""
    for i, s_i in enumerate(s):
        whole = join_strings(whole, s_i)

    return whole


def likelihood_delta(x: str, y: str, ctx: NarrativeContext):
    # Find how the presence of x before y changes the likelihood of y
    xy = join_strings(x, y)
    p_xy = p(xy, ctx)
    p_x = p(x, ctx)
    p_y = p(y, ctx)

    return {
        "p_xy": p_xy,
        "p_x": p_x,
        "p_y": p_y,
        "likelihood_delta": (p_xy - p_x) - p_y,
    }


# Precompute narrative set -- text pairs
def stratified_n_tuple_sampling(values, sample_size):
    # Create the stack
    stack = []
    counts = {}
    for value in values:
        stack.extend([value] * sample_size)
        counts[value] = sample_size

    # Shuffle the stack
    random.shuffle(stack)

    assert (
        len(stack) == len(values) * sample_size
    ), f"Stack has {len(stack)} elements, expected {len(values) * sample_size}"
    # Construct batches
    batches = []
    for _ in range(len(values)):
        batch = []
        for _ in range(sample_size):
            idx = 0
            max_count = max(counts.values())
            while stack[idx] in batch or counts[stack[idx]] != max_count:
                idx += 1

            # Pop the element
            # batch.append(stack.pop(idx))
            val = stack.pop(idx)
            batch.append(val)
            counts[val] -= 1
        #     print("Stack:", stack)
        #     print("Batch:", batch)
        # print("Appending batch:", batch)
        batches.append(batch)

    assert (
        len(stack) == 0
    ), f"Stack is not empty, has {len(stack)}/{len(values) * sample_size} elements: {stack}"

    # Assert every element has count 0
    for value in counts:
        assert counts[value] == 0, f"Value {value} has count {counts[value]}"
    return batches


def construct_pairs(labels: List[str], other_labels: List[str], ctx: NarrativeContext):
    # We need to get all permutations of size ctx.conditioners from labels, and pair each
    # with each label in other_labels
    permutations = list(itertools.permutations(labels, ctx.conditioners))

    pairs = []
    for other_label in other_labels:
        candidate_permutations = [p for p in permutations if other_label not in p]
        contains_other_label = len(candidate_permutations) != len(permutations)

        this_labels = [label for label in labels if label != other_label]
        # Determine which permutations to use
        if ctx.conditioners == 1:
            # Use all permutations
            selected_permutations = candidate_permutations
        elif ctx.sample_rate == -1:
            # Sample permutations equal to if ctx.conditioners == 1
            # selected_permutations = random.sample(
            #     candidate_permutations,
            #     (len(labels) - 1 if contains_other_label else len(labels))
            # )
            selected_permutations = stratified_n_tuple_sampling(
                this_labels,
                ctx.conditioners,
            )
        elif ctx.sample_rate == -100:
            # Use all possible choices of permutations from candidate_permutations, selecting size ctx.conditioners
            selected_permutations = candidate_permutations
        elif ctx.sample_rate > 1:
            # Assert integer sample rate (multiple of n)
            assert (
                ctx.sample_rate % 1 == 0
            ), f"Sample rate {ctx.sample_rate} is not a multiple of {ctx.conditioners}"
            # Sample ctx.sample_rate times
            selected_permutations = []
            for _ in range(int(ctx.sample_rate)):
                selected_permutations.extend(
                    stratified_n_tuple_sampling(
                        this_labels,
                        ctx.conditioners,
                    )
                )
        else:
            raise Exception("Not implemented")

        # Construct pairs
        for permutation in selected_permutations:
            # Ensure the other label is not in the permutation
            if other_label in permutation:
                raise Exception(
                    "Other label is in permutation; this should not happen at this point!"
                )

            # Add the pair
            pairs.append((permutation, other_label))

    # Return all pairs
    return pairs


def experiment_set(data: Dict[str, List[str]], ctx: NarrativeContext):
    pairs_map = {
        (label_set, other_label_set): construct_pairs(
            data[label_set], data[other_label_set], ctx
        )
        for label_set in data
        for other_label_set in data
    }

    # Find distributions of conditionals across label sets, N(mu, sigma) per pair
    distributions = {}
    pair_sets = []
    for label_set in data:
        for other_label_set in data:
            pair_sets.append((label_set, other_label_set))

    # Precompute logit sums so they're cached for each pair
    to_precompute = set()
    for label_set, other_label_set in tqdm(pair_sets):
        for x, y in pairs_map[(label_set, other_label_set)]:
            # print(x, y)
            # print(" ".join(x) + " " +  y)
            # get_logits([" ".join(x) + " " +  y], ctx.model)
            # to_precompute.add(" ".join(x) + " " +  y)
            # to_precompute.add(" ".join(x))
            to_precompute.add(join_many_strings(x))
            to_precompute.add(y)
            to_precompute.add(join_many_strings([*x, y]))

    # Precompute logits
    get_logits(list(to_precompute), ctx.model)

    # Compute likelihood deltas
    for label_set, other_label_set in tqdm(pair_sets):
        pairs = pairs_map[(label_set, other_label_set)]
        results = list(
            tqdm(
                map(
                    lambda x: {
                        "x": x[0],
                        "y": x[1],
                        **likelihood_delta(join_many_strings(x[0]), x[1], ctx),
                    },
                    pairs,
                ),
                total=len(pairs),
            )
        )

        distributions[(label_set, other_label_set)] = {
            "results": results,
            "distribution": stats.norm.fit([x["likelihood_delta"] for x in results]),
        }

    return distributions
