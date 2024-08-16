# # Parameters cell

# # Model
# MODEL_LOC = None
# MODEL = "babbage-002"
# # MODEL = "df@meta-llama/Llama-2-70b-chat-hf"

# # Conditioners
# CONDITIONERS = 1

# # Sample rate
# SAMPLE_RATE = -100

# # Debug
# DEBUG = False

# # Corpus
# CORPUS_NAME = "dog_ownership"

# # The name of the experiment (i.e. where to save the results)
# EXPERIMENT_NAME = "test_ignore_me"

# # The size of the corpus
# CORPUS_SIZE = "/20_10"


CORPORA_NAMES=(
    climate_change
    domestic_violence
    misogyny
    dog_ownership
)

# NUM_CONDITIONERS=(
#     1
# )

# MODELS=(
#     babbage-002
#     davinci-002
# )

# MODELS=(
#     lambda@TheBloke/mixtral-8x7b-v0.1-AWQ
# )

# MODELS=(
#     lambda@TheBloke/Llama-2-70B-AWQ
# )

# MODELS=(
#     together@meta-llama/Llama-2-70b-chat-hf
#     together@mistralai/Mixtral-8x7B-Instruct-v0.1
# )

NUM_CONDITIONERS=(
    1
    2
)

# MODELS=(
#     df@databricks/dbrx-instruct
# )

# MODELS=(
#     lambda@meta-llama/Meta-Llama-3-8B-Instruct
# )

# MODELS=(
#     lambda@casperhansen/llama-3-70b-instruct-awq
# )

MODELS=(
    lambda@meta-llama/Meta-Llama-3-70B
)


DATASET_TYPE=distilled

for CORPUS_NAME in "${CORPORA_NAMES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # for CONDITIONERS in 1 2; do
        for CONDITIONERS in "${NUM_CONDITIONERS[@]}"; do
            # Replace '@' with '_at_' and '/' with '_' for filenames
            MODEL_FILENAME=$(echo "$MODEL" | sed 's/@/_at_/g' | sed 's/\//_/g')

            echo "Running new method experiment for model $MODEL and corpus $CORPUS_NAME" with dataset type $DATASET_TYPE

            python3 -m papermill new_method_experiment.ipynb \
                corpus_results/notebooks/new_method_experiment_${CORPUS_NAME}_${MODEL_FILENAME}_${CONDITIONERS}_${DATASET_TYPE}.ipynb \
                -p MODEL "$MODEL" \
                -p CONDITIONERS $CONDITIONERS \
                -p SAMPLE_SIZE -1 \
                -p SAMPLE_COUNT -1 \
                -p SAMPLE_MULTIPLIER 1 \
                -p CORPUS_NAME $CORPUS_NAME \
                -p EXPERIMENT_NAME "new_method_experiment_${CORPUS_NAME}_${MODEL_FILENAME}_${CONDITIONERS}_${DATASET_TYPE}" \
                -p DEBUG False \
                -p CORPUS_SIZE "/20_10" \
                -p DATASET_TYPE "${DATASET_TYPE}" \
                --log-output
        done
    done
done
