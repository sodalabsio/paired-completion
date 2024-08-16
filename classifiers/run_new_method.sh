# CORPORA=(
#     converted_climate_change_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
#     converted_dog_ownership_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
#     converted_domestic_violence_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
#     converted_misogyny_gpt-4-turbo-preview_gpt-4-turbo-preview_10_10
# )

# MODELS=(
#     babbage-002
#     davinci-002
# )

# for CORPUS in "${CORPORA[@]}"; do
#     for MODEL in "${MODELS[@]}"; do
#         for CONDITIONERS in 1 2; do
#             echo "Running classifiers for model $MODEL and corpus $CORPUS"

#             python3 -m papermill new_method.ipynb \
#                 corpus_results/notebooks/new_method${CORPUS}_${MODEL}_${CONDITIONERS}.ipynb \
#                 -p MODEL "$MODEL" \
#                 -p CONDITIONERS $CONDITIONERS \
#                 -p SAMPLE_SIZE -1 \
#                 -p SAMPLE_COUNT -1 \
#                 -p SAMPLE_MULTIPLIER 1 \
#                 -p CORPUS $CORPUS \
#                 -p NARRATIVE_SET "climate" \
#                 -p NARRATIVE_SUFFIX "" \
#                 -p EXPERIMENT_NAME "classifiers_${CORPUS}_${MODEL}_${CONDITIONERS}" \
#                 -p DEBUG False
#         done
#     done
# done
