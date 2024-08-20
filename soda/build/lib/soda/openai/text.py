import os
import json
import time
import subprocess
import itertools
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Tuple, Dict
from joblib import Parallel, delayed


def get_model_client_headers(model, kwargs):
    # Use OpenAI for babbage-002 and davinci-002
    if model in ("babbage-002", "davinci-002"):
        headers = {}
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        return model, client, headers

    # Use DeepInfra for other models
    headers = {}
    client = OpenAI(
        api_key=os.environ.get("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
    )
    return model, client, headers


def get_vertex_ai_bearer(kwargs):
    if "bearer" in kwargs:
        b = kwargs["bearer"]
        del kwargs["bearer"]
        return b

    return (
        subprocess.check_output(["gcloud", "auth", "print-identity-token"])
        .decode("utf-8")
        .strip()
    )


def completion_model(x: Union[str, List[str]], model: str, **kwargs):
    # If x is a list and has one element, use the string API instead
    if isinstance(x, list) and len(x) == 1:
        x = x[0]

    checkKwargs(kwargs)

    model, client, headers = get_model_client_headers(model, kwargs)

    for i in range(10):
        try:
            print("Prompt:", x)
            print("Model:", model)
            result = client.completions.create(
                model=model, prompt=x, extra_headers=headers, **kwargs
            )

            if not result:
                raise Exception("No response")

            return result
        except Exception as e:
            # print(e)
            print("Backing off (i = {}, time = {}, error = {})".format(i, 1.2**i, e))
            time.sleep(1 + 1.2**i)
            pass

    raise Exception("Failed to call model")


def completion_model_batched(
    x: List[str], model: str, batch_size: int = 20, num_threads: int = 16, **kwargs
):
    if batch_size != 1:
        return completion_model_batched_openai(
            x, model, batch_size, num_threads, **kwargs
        )

    # Using ThreadPoolExecutor to parallelize the API calls, ensuring we return in the same order
    # as the input list

    def call_completion_model(i, prompt):
        return i, completion_model(prompt, model, **kwargs)

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all the tasks and get a list of Future objects
        future_to_prompt = {
            executor.submit(call_completion_model, i, prompt): prompt
            for i, prompt in enumerate(x)
        }

        for future in tqdm(as_completed(future_to_prompt), total=len(future_to_prompt)):
            results.append(future.result())

    # Sort the results by the original order of the prompts
    results.sort(key=lambda x: x[0])
    results = [x[1] for x in results]

    # Flatten results
    results = list(itertools.chain.from_iterable(results))

    return results


def completion_model_batched_openai(
    x: List[str], model: str, batch_size: int = 20, num_threads: int = 16, **kwargs
):
    if len(x) == 0:
        raise Exception("Empty input")

    # Split the list into chunks of max size = batch_size, and run them one by one
    chunks = []
    for i in range(0, len(x), batch_size):
        chunks.append(x[i : i + batch_size])

    # Iterative version
    # result_obj = None
    # for chunk in tqdm(chunks):
    #     if result_obj is None:
    #         result_obj = completion_model(chunk, model, **kwargs)
    #     else:
    #         result_obj.choices.extend(completion_model(chunk, model, **kwargs).choices)

    # Threaded version
    def call_model_batch(batch: List[str]):
        return completion_model(batch, model, **kwargs)

    results = Parallel(n_jobs=num_threads, backend="threading")(
        delayed(call_model_batch)(batch) for batch in tqdm(chunks)
    )
    if results is None:
        raise Exception("No results")

    result_obj = results[0]
    for result in results[1:]:
        result_obj.choices.extend(result.choices)

    assert len(result_obj.choices) == len(x), "Something went wrong with batching"
    return result_obj


def instruct_chat_model(
    system: str,
    prompt: str,
    model: str,
    messages: Union[List[Dict[str, str]], None] = None,
    examples: Union[List[Dict[str, str]], None] = None,
    **kwargs,
):
    checkKwargs(kwargs)

    assert messages or (system and prompt), "Must provide messages or system and prompt"
    assert not (
        messages and (system or prompt)
    ), "Must provide messages or system and prompt, not both"
    assert model is not None, "Must provide model"

    if not messages:
        messages = [
            {"role": "system", "content": system},
        ]

        if examples:
            for x, y in examples:
                messages.append({"role": "user", "content": x})
                messages.append({"role": "assistant", "content": y})
        messages.append({"role": "user", "content": prompt})

        model, client, headers = get_model_client_headers(model, kwargs)

        try:
            result = client.chat.completions.create(
                model=model, messages=messages, extra_headers=headers, **kwargs
            )
        except Exception as e:
            print("*** Failed on this message ***")
            print(json.dumps(messages, indent=2))
            raise e

        if not result:
            raise Exception("No response")

        return result

    raise Exception("Failed to call model")


def checkKwargs(kwargs):
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 4096


def instruct_chat_model_batched(
    system: str,
    prompts: List[str],
    model: str,
    examples: Union[List[Tuple[str, str]], None] = None,
    num_threads=16,
    **kwargs,
):
    # Define a function to call instruct_chat_model on each prompt
    def call_instruct_chat_model(i, prompt):
        return i, instruct_chat_model(system, prompt, model, examples, **kwargs)

    # Use ThreadPoolExecutor to parallelize the API calls
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all the tasks and get a list of Future objects
        future_to_prompt = {
            executor.submit(call_instruct_chat_model, i, prompt): prompt
            for i, prompt in enumerate(prompts)
        }

        for future in tqdm(as_completed(future_to_prompt), total=len(future_to_prompt)):
            results.append(future.result())

    # Sort the results by the original order of the prompts
    results.sort(key=lambda x: x[0])
    results = [x[1] for x in results]

    return results
