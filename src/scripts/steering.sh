

steering_dataset="stereoset"  # support: truthfulqa, toxigen, confaide, stereoset, sst-2, pku-rlhf-10k
eval_dataset="stereoset"  # support: truthfulqa, toxigen, confaide, stereoset, sst-2
save_dir="your path"
target_model='AmberChat'
ckpt_name="ckpt_179"

# 1. download the checkpoints
huggingface-cli download LLM360/Amber --revision $ckpt_name --local-dir $save_dir/$ckpt_name --local-dir-use-symlinks False 
huggingface-cli download LLM360/AmberChat --local-dir $save_dir/AmberChat --local-dir-use-symlinks False 

# 2. collect the activations
python generate_activations.py --model $ckpt_name --layers -1 --dataset $steering_dataset 

# 3. generate the steering vector
python generate_steering_vector.py --model $ckpt_name --layer_list -1 --dataset $steering_dataset 


# 4. inference intervention with steering vector
python eval_trustworthiness.py --model $target_model --dataset $eval_dataset --from_model $ckpt_name --from_dataset $steering_dataset --layer_list 16 --alpha_list -1

