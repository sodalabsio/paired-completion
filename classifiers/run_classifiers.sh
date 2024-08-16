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

# for CORPUS in "${CORPORA[@]}"

for CORPUS_NAME in "${CORPORA_NAMES[@]}";
do
    echo "Running classifiers for $CORPUS_NAME"
    python3 -m papermill classifiers.ipynb \
        corpus_results/notebooks/classifiers_${CORPUS_NAME}.ipynb \
        -p CORPUS_NAME $CORPUS_NAME \
        -p EXPERIMENT_NAME "classifiers_${CORPUS_NAME}" \
        -p CORPUS_SIZE "/20_10" \
        --log-output
done