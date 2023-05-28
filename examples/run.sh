TASK=cls
DATASET=trec
#TASK=ner
#DATASET=conll2003
MODEL=distilbert
OUTPUT_DIR=output
# keep datasets up to 10GB in memory
export HF_DATASETS_IN_MEMORY_MAX_SIZE=1e10
export CUDA_VISIBLE_DEVICES=0 # multi-heads currently only support single gpu

# data preparation stage
# runs the data preparation pipeline specified in the
# configuration file and saves the prepared dataset
# to the specified location
python -m hyped.stages.prepare \
    -c ./$TASK/$DATASET/${MODEL}_data.json \
    -o $OUTPUT_DIR/$TASK/$DATASET/${MODEL}_data

# model training stage
# trains a model on the prepared data generated by the
# previous stage
python -m hyped.stages.train \
    -c ./$TASK/$DATASET/${MODEL}_run.json \
    -d $OUTPUT_DIR/$TASK/$DATASET/${MODEL}_data \
    -o $OUTPUT_DIR/$TASK/$DATASET/$MODEL

# model evaluation stage
# evaluates the trained model on the test split of the
# prepared dataset generated earlier
python -m hyped.stages.test \
    -c ./$TASK/$DATASET/${MODEL}_run.json \
    -d $OUTPUT_DIR/$TASK/$DATASET/${MODEL}_data \
    -m $OUTPUT_DIR/$TASK/$DATASET/$MODEL/best-model
