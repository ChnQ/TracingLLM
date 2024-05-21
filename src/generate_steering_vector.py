import os
import torch
import argparse
import pandas as pd
from generate_activations import load_acts

def get_steering_vector(dataset_name, model_tag, layer, labels, device='cuda:0', output_dir='activations', train_ratio=0.5):

    acts = load_acts(dataset_name, model_tag, layer=layer, center=False, scale=False, device=device, acts_dir=output_dir)
    full_acts = acts.clone().detach()

    # using training set to generate steering vector
    train_num = int(len(labels) * train_ratio)
    acts = acts[:train_num]
    labels = labels[:train_num]
        
    true_mass_mean = torch.mean(acts[labels == 1], dim=0)
    false_mass_mean = torch.mean(acts[labels == 0], dim=0)
    direction_vector = true_mass_mean - false_mass_mean
    direction_vector = direction_vector / direction_vector.norm()

    proj_vals = full_acts @ direction_vector
    proj_val_std = torch.std(proj_vals)
    direction_vector = proj_val_std * direction_vector

    return direction_vector


if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="AmberChat",
                        help="Size of the model to use. Options are 7B or 30B")
    parser.add_argument('--layer_list', nargs='+', type=int, default=-1)
    parser.add_argument("--dataset", default='truthfulqa',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="activations",
                        help="Directory to save activations to")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_ratio", default=0.5)
    args = parser.parse_args()

    model_tag = args.model
    dataset_name = args.dataset

    if dataset_name == 'truthfulqa':
        dataset_name = 'truthfulqa_train'
        args.train_ratio = 1
    
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    labels = dataset['label'].to_numpy()

    save_dir = f'steering_vectors/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    layers = [int(layer) for layer in args.layer_list]
    if layers == [-1]:
        layers = list(range(32))
    for layer in layers:
        direction_vector = get_steering_vector(dataset_name, model_tag, layer, labels, device=args.device, train_ratio=args.train_ratio)
        torch.save(direction_vector, f'{save_dir}/{model_tag}_layer{layer}.pt')
        print(f'Steering vector of layer {layer} has been saved to {save_dir}/{model_tag}_layer{layer}.pt!')