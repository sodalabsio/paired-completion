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

MODELS=(
    babbage-002
    davinci-002
)

NUM_CONDITIONERS=(
    1
    2
)

for CORPUS in "${CORPORA_NAMES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # for CONDITIONERS in 1 2; do
        for CONDITIONERS in "${NUM_CONDITIONERS[@]}"; do
            echo "Running classifiers for model $MODEL and corpus $CORPUS"

            python3 -m papermill narrative_analysis.ipynb \
                corpus_results/notebooks/narrative_analysis_${CORPUS}_${MODEL}_${CONDITIONERS}.ipynb \
                -p MODEL "$MODEL" \
                -p CONDITIONERS $CONDITIONERS \
                -p SAMPLE_RATE -100 \
                -p DEBUG False \
                -p CORPUS_NAME $CORPUS \
                -p EXPERIMENT_NAME "narrative_analysis_${CORPUS}_${MODEL}_${CONDITIONERS}" \
                -p CORPUS_SIZE "/20_10"
        done
    done
done
