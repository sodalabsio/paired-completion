# CORPORA_NAMES=(
#     climate_change/seed_mistral@mistral-large-latest_sentence_mistral@mistral-large-latest_temp_0.0
#     domestic_violence/seed_mistral@mistral-large-latest_sentence_mistral@mistral-large-latest_temp_0.0
#     misogyny/seed_mistral@mistral-large-latest_sentence_mistral@mistral-large-latest_temp_0.0
# )

CORPORA_NAMES=(
    dog_ownership/seed_mistral@mistral-large-latest_sentence_mistral@mistral-large-latest_temp_0.0
)
#    dog_ownership
# CORPORA_NAMES=(
#     dog_ownership
# )

#    climate_change
#    dog_ownership
#    domestic_violence
#    misogyny
# climate_change
# converted_dog_ownership_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
# converted_domestic_violence_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
# converted_misogyny_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10

# MODELS=(
#     gpt-3.5-turbo
#     gpt-4-turbo-preview
#     together@meta-llama/Llama-2-70b-chat-hf
#     together@mistralai/Mixtral-8x7B-Instruct-v0.1
# )

MODELS=(
    together@meta-llama/Llama-3-8b-chat-hf
    together@meta-llama/Llama-3-70b-chat-hf
    together@microsoft/phi-2
    together@mistralai/Mixtral-8x22B-Instruct-v0.1
)
#    gpt-3.5-turbo
#    gpt-4-turbo-preview
#    together@meta-llama/Llama-2-70b-chat-hf
#    together@mistralai/Mistral-7B-Instruct-v0.1
#    together@mistralai/Mistral-7B-Instruct-v0.2
#    together@meta-llama/Llama-2-7b-chat-hf
#    together@meta-llama/Llama-2-13b-chat-hf

# gpt-3.5-turbo-0125

MODES=(
    seeds
    distilled
    summaries
    zero-shot
)

for CORPUS_NAME in "${CORPORA_NAMES[@]}"; do
    for MODE in "${MODES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "Running classifiers for model $MODEL and corpus $CORPUS with mode $MODE."

            python3 -m papermill gpt.ipynb \
                corpus_results/gpt_${CORPUS_NAME//\//-}_${MODEL//\//-}.ipynb \
                -p CORPUS_NAME $CORPUS_NAME \
                -p EXPERIMENT_NAME "${CORPUS_NAME//\//-}_${MODEL//\//-}" \
                -p MODEL "$MODEL" \
                -p MODE "$MODE" \
                -p CORPUS_SIZE "/N_20_K_10" \
                --log-output
        done
    done
done
