import os
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from generate_activations import load_acts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mi_estimators import estimate_mi_hsic

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="AmberChat",
                        help="Name of model")
    parser.add_argument("--layers", nargs='+', type=int, default=-1,
                        help="Layers to save embeddings from, -1 denotes all layers")
    parser.add_argument("--dataset", default="stereoset",
                        help="Names of dataset, without .csv extension")
    parser.add_argument("--output_dir", default="activations",
                        help="Directory to save activations to")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_tag = args.model
    device = args.device

    dataset_name = args.dataset

    save_dir = f"results/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)


    ## --------------- 1. load representations  --------------
    all_layer_acts = []
    layers = [int(layer) for layer in args.layers]
    if layers == [-1]:
        layers = list(range(32))  # 32 layers for AmberChat


    for layer in layers:    
        acts = load_acts(dataset_name, model_tag, layer=layer, center=True, scale=True, device=args.device, acts_dir=args.output_dir)
        all_layer_acts.append(acts)

    all_layer_acts = torch.stack(all_layer_acts).to(args.device)

    ## --------------- 2. params  --------------
    # params
    num_layers = all_layer_acts.shape[0]
    num_prompts = all_layer_acts.shape[1]
    hidden_dim = all_layer_acts.shape[2]
    EPOCH = 1000
    train_ratio = 0.8  
    num_train = int(num_prompts * train_ratio)
    num_test = num_prompts - num_train

    ## --------------- 3. train probe and calc MI --------------- 
    targets = pd.read_csv(f"datasets/{dataset_name}.csv", dtype={'label': int})['label'].to_numpy()
    targets = torch.tensor(targets, device=args.device, dtype=torch.float32)
    targets = targets.unsqueeze(0).repeat(num_layers, 1)

    # train/val splitting
    train_activations = all_layer_acts[:, :num_train, :]
    test_activations = all_layer_acts[:, num_train:, :]
    train_targets = targets[:, :num_train]
    test_targets = targets[:, num_train:]

    acc_list = []
    # save the list of mutual information
    MI_tx = [] # I(X;Y)
    MI_ty = [] # I(T;Y)
    for idx, layer in enumerate(layers):

        train_set = train_activations[idx].cpu().numpy()
        train_label = train_targets[idx].cpu().numpy()
        test_set = test_activations[idx].cpu().numpy()
        test_label = test_targets[idx].cpu().numpy()


        ''' record the first layer representation as X to compute I(X; T), where T is the embedding '''
        if layer == 0:
            X = train_set
            Y = train_label

        ''' sklearn '''
        probe = LogisticRegression()
        probe.fit(train_set, train_label)

        y_prob = np.max(probe.predict_proba(test_set), axis=1)
        y_pred = probe.predict(test_set)
        acc = accuracy_score(test_label, y_pred)
        acc_list.append(acc)
        print('Layer {}, ACC {}'.format(layers[idx], acc))

        ''' information computation using HSIC '''
        IXT, IYT = estimate_mi_hsic(X, Y, train_set) 
        print(f"I(X;T): {IXT}")
        print(f"I(Y;T): {IYT}")
        print("-----------------------------------------------")
        MI_tx.append(IXT)
        MI_ty.append(IYT)

        del probe

    # save results
    np.save(f'{save_dir}{model_tag}_acc.npy', acc_list)
    np.save(f'{save_dir}{model_tag}_tx.npy', MI_tx)
    np.save(f'{save_dir}{model_tag}_ty.npy', MI_ty)