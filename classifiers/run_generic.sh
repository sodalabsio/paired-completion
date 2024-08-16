# CORPORA_NAMES=(
#     artificial_intelligence
#     renewable_energy
#     telecommuting
#     prison_reform
# )

CORPORA_NAMES=(
    artificial_meat
    big_tech_monopolies
    climate_engineering
    genetic_engineering
    gun_control
    homelessness
    immigration_policy
    net_neutrality
    universal_basic_income
    universal_voting
    vaccination
    vaccination_passports
)

# NUM_CONDITIONERS=(
#     1
#     2
# )

NUM_CONDITIONERS=(
    2
)

MODELS=(
    lambda_1x@TheBloke/Llama-2-70B-AWQ
)

MODES=(
    seeds
    sentences
)

for MODE in "${MODES[@]}"; do
    for CORPUS_NAME in "${CORPORA_NAMES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            # for CONDITIONERS in 1 2; do
            for CONDITIONERS in "${NUM_CONDITIONERS[@]}"; do
                # Replace '@' with '_at_' and '/' with '_' for filenames
                MODEL_FILENAME=$(echo "$MODEL" | sed 's/@/_at_/g' | sed 's/\//_/g')

                echo "Running new method experiment for model $MODEL and corpus $CORPUS_NAME" with dataset type $DATASET_TYPE

                python3 -m papermill new_method_experiment-generic-dataset.ipynb \
                    corpus_results/notebooks/new_method_experiment-generic-dataset_${CORPUS_NAME}_${MODEL_FILENAME}_${CONDITIONERS}_${DATASET_TYPE}.ipynb \
                    -p MODEL "$MODEL" \
                    -p CONDITIONERS $CONDITIONERS \
                    -p SAMPLE_SIZE -1 \
                    -p SAMPLE_COUNT -1 \
                    -p SAMPLE_MULTIPLIER 1 \
                    -p CORPUS_NAME $CORPUS_NAME \
                    -p EXPERIMENT_NAME "new_method_experiment_${CORPUS_NAME}_${MODEL_FILENAME}_${CONDITIONERS}_${DATASET_TYPE}_${MODE}" \
                    -p DEBUG False \
                    -p MODE "${MODE}" \
                    -p CORPUS_SIZE "/20_10" \
                    --log-output
            done
        done
    done
done
