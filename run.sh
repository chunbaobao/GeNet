
MODELS=("MLP" "GAT" "GatedGCN" "GCN")

DATASETS=("cifar10" "mnist")


OUTPUT="./out"


for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        python train.py --out $OUTPUT --model_name $MODEL --dataset_name $DATASET
    done
done