import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import json
import time
import os
import pdb
import torch
import argparse
import random
import itertools
from scipy.linalg import fractional_matrix_power
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics

from pairs.models.model import Model
from pairs.utils.data import sample_user, get_pref, to_preference, to_preference_prism, get_pref_prism, get_users_enums
from pairs.utils.inference import get_embeddings, sigmoid, sigmoid_derivative, MLE_user_weight, choose_personalized_response, max_norm_vector


import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train preference learning model')
    
    # Add arguments
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--method', type=str, default='max_dist', help='Method to use for discovery')
    parser.add_argument('--dir_name', type=str, default='debug', help='Output directory name')
    parser.add_argument('--dataset', type=str, default='PRISM', help='Dataset to use, options are synthetic or PRISM')
    parser.add_argument('--num_test_pairs', type=int, default=100, help='Number of test pairs')
    parser.add_argument('--model_dir', type=str, default='', help='Directory of the trained base reward functions model')
    return parser.parse_args()


def compute_test_metrics(u, H_test, y_test):
    """
    Computes the test loss and accuracy for the test data.

    Returns:
    - loss_value: Scalar value of the test loss
    - auc: Scalar value of AUC ROC
    """
    H_test = torch.tensor(H_test).float().cuda()
    y_test = torch.tensor(y_test).float().cuda()
    u = u.cuda()
    with torch.no_grad():
        logits = H_test @ u  # Shape: (N,)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_test.squeeze())
        
        # Compute probabilities
        probs = torch.sigmoid(logits)

    y = y_test.squeeze().cpu().detach().numpy()
    pred = probs.cpu().detach().numpy()[y != 0.5]
    y = y[y != 0.5]
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc_score = metrics.auc(fpr, tpr)

    # find the threshold that maximizes accuracy
    max_accuracy = 0
    max_threshold = 0
    for threshold in thresholds:
        binarized_pred = np.array(pred) > threshold
        accuracy = np.mean(y == binarized_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_threshold = threshold
    
    return loss.item(), auc_score, max_accuracy

def training(
        user, 
        embeddings, 
        test_embeddings, 
        test_preferences_probs, 
        data_df,
        dfs_responses,
        base_responses,
        dfs_embeddings,
        dataset,
        num_steps=40, 
        method="max_dist", 
        user_init=None, 
        use_deriviative_in_norm=True, 
        use_deriviative_in_sigma=True
    ):

    candidates = list(range(embeddings.shape[0]))
    embeddings_numpy = embeddings.float().cpu().detach().numpy().copy()
    train_loss = []
    val_loss = []
    roc_metric = []
    max_accuracy_metric = []
    preference_eval_results = []
    
    Sigma = 0.001*np.eye(embeddings_numpy.shape[1])

    item_embeddings = []
    item_preferences = []
    for i in tqdm(range(num_steps)):
        if method == "random" or i == 0:
            next_item_idx = random.choice(candidates)
        elif method == "max_dist":
            Sigma = 0.001*np.eye(embeddings_numpy.shape[1])
            for vec in item_embeddings:
                if use_deriviative_in_sigma:
                    Sigma = Sigma + sigmoid_derivative(curr_user_embedding.numpy().T @ vec) * vec[:, np.newaxis] @ vec[:, np.newaxis].T
                else:
                    Sigma = Sigma + vec[:, np.newaxis] @ vec[:, np.newaxis].T
            next_item_idx = max_norm_vector(candidates, embeddings_numpy, Sigma, curr_user_embedding, use_derivitaive=use_deriviative_in_norm)
        candidates.remove(next_item_idx)
        
        next_item_embedding = embeddings_numpy[next_item_idx, :]
        pref1, pref2 = get_pref_prism((data_df['instruction'][next_item_idx], data_df['response_1'][next_item_idx], data_df['response_2'][next_item_idx], user))
        next_item_preference = to_preference_prism(pref1, pref2)
        item_embeddings.append(next_item_embedding)
        item_preferences.append(next_item_preference)
        
        curr_user_embedding, loss = MLE_user_weight(item_embeddings, item_preferences, user_initial=user_init)
        train_loss.append(loss)
        
        v_loss, roc, max_accuracy = compute_test_metrics(curr_user_embedding, test_embeddings, test_preferences_probs)
        val_loss.append(v_loss)
        roc_metric.append(roc)
        max_accuracy_metric.append(max_accuracy)

        # Perform preference evaluation every 5 steps
        if (i % 5 == 0 and i != 0) or i == num_steps - 1:
            step_results = []
            for j in range(len(dfs_responses)):
                responses_dataframe = dfs_responses[j]
                chosen_response = choose_personalized_response(curr_user_embedding.detach().numpy(), dfs_embeddings[j].float().cpu().detach().numpy(), responses_dataframe)
                if dataset == 'synthetic':
                    p = get_pref((responses_dataframe['instruction'][0], chosen_response, base_responses[j], user))
                    p_int = to_preference(p)
                elif dataset == 'PRISM':
                    p = get_pref_prism((responses_dataframe['instruction'][0], chosen_response, base_responses[j], user))
                    p_int = to_preference_prism(p[0], p[1])
                step_results.append(p_int)
            preference_eval_results.append({
                'step': i,
                'mean_score': np.mean(step_results),
                'results': step_results
            })

    return train_loss, val_loss, roc_metric, max_accuracy_metric, curr_user_embedding, preference_eval_results


def main():
    args = parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    output_dir = os.path.join('output/discover_user', args.dir_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)

    model_dir = args.model_dir

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    model = Model(config['n_pairs'], config['n_users'], config['feature_dim'], normalize=False).cuda()
    model.device = model.model.device
    model_path = os.path.join(model_dir, "model.pth")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    if args.dataset == "PRISM":
        data_path = "data/PRISM/prompts_and_responses_test_jailbreak.csv"
        df = pd.read_csv(data_path)
        df = df.rename(columns={'prompt': 'instruction', 'original_text': 'response_1', 'revised_text': 'response_2'})
        data_df = df

        dfs_responses = []
        base_responses = []
        eval_prompts_indices = list(range(50))
        for i in eval_prompts_indices:
            data_path = f"data/PRISM/evaluation/PRISM_evaluation_data_{i}.csv"
            base_response_path = f"data/PRISM/evaluation/PRISM_evaluation_data_{i}_base_response.txt"
            temp_df = pd.read_csv(data_path)
            dfs_responses.append(temp_df)
            with open(base_response_path, "r") as f:
                base_responses.append(f.read())
        test_embeddings, train_embeddings, dfs_embeddings = get_embeddings(data_df, dfs_responses, base_responses, model_dir, model, num_test_pairs=args.num_test_pairs)
        persona_dataset_test = load_from_disk("data/PRISM/prism_personas_test.hf")
        persona_dataset_train = load_from_disk("data/PRISM/prism_personas_train.hf")
        users = persona_dataset_train['persona_description'] + persona_dataset_test['persona_description']
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    user = sample_user(model_dir, users)[0]

    test_preferences = []
    for i in tqdm(range(len(data_df)-args.num_test_pairs, len(data_df))):
        if args.dataset == 'synthetic':
            p = get_pref((data_df['instruction'][i], data_df['response_1'][i], data_df['response_2'][i], user))
        elif args.dataset == 'PRISM':
            p = get_pref_prism((data_df['instruction'][i], data_df['response_1'][i], data_df['response_2'][i], user))
        test_preferences.append(p)
    if args.dataset == 'synthetic':
        test_preferences_probs = np.array(list(map(to_preference, test_preferences)))
    elif args.dataset == 'PRISM':
        test_preferences_probs = np.array([to_preference_prism(pref1, pref2) for pref1, pref2 in test_preferences])
    test_preferences_probs = torch.tensor(np.array(test_preferences_probs)).cuda()

    if args.method == "random":
        train_loss, val_loss, roc_metric, max_accuracy_metric, learned_embedding, preference_eval_results = training(
            user, train_embeddings, test_embeddings, test_preferences_probs, data_df, 
            dfs_responses, base_responses, dfs_embeddings, args.dataset, method="random"
        )
    elif args.method == "max_dist":
        train_loss, val_loss, roc_metric, max_accuracy_metric, learned_embedding, preference_eval_results = training(
            user, train_embeddings, test_embeddings, test_preferences_probs, data_df,
            dfs_responses, base_responses, dfs_embeddings, args.dataset, method="max_dist",
            use_deriviative_in_norm=True, use_deriviative_in_sigma=True
        )
    print(learned_embedding)
    print(train_loss)
    print(val_loss)
    print(roc_metric)
    print(max_accuracy_metric)

    results_dict = {
        "learned_embedding": learned_embedding.tolist(),
        "train_loss": np.array(train_loss).tolist(),
        "val_loss": np.array(val_loss).tolist(),
        "roc_metric": np.array(roc_metric).tolist(),
        "max_accuracy_metric": np.array(max_accuracy_metric).tolist(),
        "preference_eval_results": preference_eval_results,
    }
    with open(os.path.join(output_dir, f"results_{args.method}_seed_{args.random_seed}.json"), "w") as f:
        json.dump(results_dict, f)

if __name__ == "__main__":
    main()