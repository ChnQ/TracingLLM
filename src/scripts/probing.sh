

model_list=(
    ckpt_000
    ckpt_001
    ckpt_002
    ckpt_005
    ckpt_010
    ckpt_020
    ckpt_050
    ckpt_100
    ckpt_179
    ckpt_358
)

dataset="stereoset"  # support: toxigen, confaide, stereoset, sst-2
save_dir="your path"

for model in "${model_list[@]}"; do
    # 1. download the checkpoints
    model_path=$save_dir/$model
    if [ ! -d "$model_path" ]; then
        echo "$model is downloading..."
        huggingface-cli download LLM360/Amber --revision $model --local-dir $model_path --local-dir-use-symlinks False 
    else
        echo "$model is downloaded!"
    fi

    # 2. collect the activations
    python generate_activations.py --model $model --layers -1 --dataset $dataset 

    # 3. train the linear probes and calculate mutual information
    python train_probes.py --model $model --dataset $dataset --layers -1

    # 4. delete the checkpoints
    echo "delete $model_path" 
    rm -rf $model_path
    rm -rf ~/.cache/huggingface/hub/models--LLM360--Amber
    
done