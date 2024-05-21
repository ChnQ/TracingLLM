# Adapted and modifed from https://github.com/saprmarks/geometry-of-truth
import os
import torch
import argparse
import pandas as pd
import configparser
from glob import glob
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

ACTS_BATCH_SIZE = 400
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        # self.out, _ = module_outputs
        self.out = module_outputs  

def load_model(model_size, device, model_tag='AmberChat'):
    model_path = f'your save path/{model_tag}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if model_size == '13B' and device != 'cpu':
        model = model.half()
    model.to(device)
    return tokenizer, model


def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements


def get_acts(statements, tokenizer, model, layers, device, token_pos=-1):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.

    token_pos: default to fetch the last token's activations
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    
    # get activations
    acts = {layer : [] for layer in layers}
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            # print(type(hook.out))
            # print(hook.out[0][0, token_pos])
            # acts[layer].append(hook.out[0, token_pos])
            acts[layer].append(hook.out[0][0, token_pos])
    
    # stack len(statements)'s activations
    for layer, act in acts.items():
        acts[layer] = torch.stack(act).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts


def load_acts(dataset_name, model_tag, layer, center=True, scale=False, device='cpu', acts_dir='activations'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(PROJECT_ROOT, acts_dir, model_tag, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    acts = [torch.load(os.path.join(directory, f'layer_{layer}_{i}.pt')).to(device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
    acts = torch.cat(acts, dim=0).to(device)
    if center:
        acts = acts - torch.mean(acts, dim=0)
    if scale:
        acts = acts / torch.std(acts, dim=0)
    return acts


if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="llama-2-7b-chat-hf",
                        help="Name of model")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="activations",
                        help="Directory to save activations to")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_tag = args.model

    torch.set_grad_enabled(False)

    ### generate acts
    
    tokenizer, model = load_model(args.model, args.device, model_tag=model_tag)

    for dataset in args.datasets:
        if dataset == 'truthfulqa':
            dataset = 'truthfulqa_train'
        statements = load_statements(dataset)
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))

        save_dir = f"{args.output_dir}/{model_tag}/{dataset}/"
        os.makedirs(save_dir, exist_ok=True)

        # reduce the load of each file
        for idx in range(0, len(statements), ACTS_BATCH_SIZE):
            acts = get_acts(statements[idx:idx + ACTS_BATCH_SIZE], tokenizer, model, layers, args.device)
            for layer, act in acts.items():
                torch.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")
