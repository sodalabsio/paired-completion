# CORPORA=(
#     converted_climate_change_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
#     # converted_dog_ownership_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
#     # converted_domestic_violence_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
#     # converted_misogyny_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
# )
CORPORA_NAMES=(
    climate_change
    domestic_violence
    misogyny
    dog_ownership
)

MODELS=(
    fasttext
    text-embedding-3-small
    text-embedding-3-large
)

# for CORPUS in "${CORPORA[@]}"

for CORPUS_NAME in "${CORPORA_NAMES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Running classifiers for $CORPUS_NAME"
        python3 -m papermill embedding.ipynb \
            corpus_results/notebooks/embedding_${CORPUS_NAME}.ipynb \
            -p CORPUS_NAME $CORPUS_NAME \
            -p EXPERIMENT_NAME "embedding_${CORPUS_NAME}_${MODEL}" \
            -p CORPUS_SIZE "/20_10" \
            -p MODEL "$MODEL" \
            --log-output
    done
done