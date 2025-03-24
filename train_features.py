import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import time
import os
import pdb
import torch
import json
import random
import itertools
import argparse
from sklearn.model_selection import train_test_split
from scipy.linalg import fractional_matrix_power
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.metrics import roc_curve, roc_auc_score
from pairs.models.model import Model
from pairs.utils.training import (
    to_preference,
    get_batch,
    plot_roc,
    calculate_loss_MLE,
    SVD_for_sparse_matrix,
)
from pairs.utils.data import get_users_enums, parse_attribute_entries
from configs.config import Config

from accelerate import Accelerator
from sklearn.metrics import roc_curve, auc

# Set deterministic CUDA behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def prepare_prism_data(df_train, df_validation, df_calibration, df_test, config):
    # Get all unique prompts and create a consistent mapping
    all_prompts = pd.concat(
        [
            df_train[["prompt", "response_1", "response_2"]].drop_duplicates(),
            df_validation[["prompt", "response_1", "response_2"]].drop_duplicates(),
            df_calibration[["prompt", "response_1", "response_2"]].drop_duplicates(),
            df_test[["prompt", "response_1", "response_2"]].drop_duplicates(),
        ]
    ).drop_duplicates()

    # Create a mapping with an explicit counter
    prompt_to_idx = {tuple(row): idx for idx, row in enumerate(all_prompts.values)}

    # Shift persona_index for calibration and test data by +1200
    df_calibration["persona_index"] = df_calibration["persona_index"] + 1200
    df_test["persona_index"] = df_test["persona_index"] + 1200

    # Function to get prompt index
    def get_prompt_idx(row):
        key = (row["prompt"], row["response_1"], row["response_2"])
        return prompt_to_idx[key]

    # Convert preferences to tensor format by mapping to consistent indices
    df_train_dedup = df_train.drop_duplicates(
        ["prompt", "response_1", "response_2", "persona_index"]
    )
    df_validation_dedup = df_validation.drop_duplicates(
        ["prompt", "response_1", "response_2", "persona_index"]
    )
    df_calibration_dedup = df_calibration.drop_duplicates(
        ["prompt", "response_1", "response_2", "persona_index"]
    )
    df_test_dedup = df_test.drop_duplicates(
        ["prompt", "response_1", "response_2", "persona_index"]
    )
    train_matrix = pd.pivot_table(
        df_train_dedup,
        values="preference",
        index=df_train_dedup.apply(get_prompt_idx, axis=1),
        columns="persona_index",
    )
    val_matrix = pd.pivot_table(
        df_validation_dedup,
        values="preference",
        index=df_validation_dedup.apply(get_prompt_idx, axis=1),
        columns="persona_index",
    )
    calibration_matrix = pd.pivot_table(
        df_calibration_dedup,
        values="preference",
        index=df_calibration_dedup.apply(get_prompt_idx, axis=1),
        columns="persona_index",
    )
    test_matrix = pd.pivot_table(
        df_test_dedup,
        values="preference",
        index=df_test_dedup.apply(get_prompt_idx, axis=1),
        columns="persona_index",
    )

    # With probability config.noise, flip the preference
    if config.noise > 0:
        train_matrix = train_matrix.applymap(
            lambda x: (1 - x if x in [0, 1] and random.random() < config.noise else x)
        )
        calibration_matrix = calibration_matrix.applymap(
            lambda x: (1 - x if x in [0, 1] and random.random() < config.noise else x)
        )

    # Combine all matrices and convert to tensor
    # First create a complete matrix with all indices and columns
    all_indices = sorted(
        list(
            set(train_matrix.index)
            | set(val_matrix.index)
            | set(calibration_matrix.index)
            | set(test_matrix.index)
        )
    )
    all_columns = sorted(
        list(
            set(train_matrix.columns)
            | set(val_matrix.columns)
            | set(calibration_matrix.columns)
            | set(test_matrix.columns)
        )
    )

    # Initialize empty matrix with all indices and columns
    preference_observations = pd.DataFrame(index=all_indices, columns=all_columns)

    # Fill in values from each matrix
    preference_observations.update(train_matrix)
    preference_observations.update(val_matrix)
    preference_observations.update(calibration_matrix)
    preference_observations.update(test_matrix)

    # Convert to tensor
    preference_observations = preference_observations.to_numpy()

    # Extract pairs and instructions in the same order as preference_observations
    pairs = []
    instructions = []
    for idx in range(len(all_prompts)):
        row = all_prompts.iloc[idx]
        pairs.append((row["response_1"], row["response_2"]))
        instructions.append(row["prompt"])

    # Extract users based on dataset persona indices and data
    max_persona_idx = max(
        df_train["persona_index"].max(),
        df_validation["persona_index"].max(),
        df_calibration["persona_index"].max(),
        df_test["persona_index"].max(),
    )
    users = []
    for i in range(max_persona_idx + 1):
        # Find persona data for this index across all datasets
        persona_data = None
        for df in [df_train, df_validation, df_calibration, df_test]:
            matches = df[df["persona_index"] == i]["persona"]
            if not matches.empty:
                persona_data = matches.iloc[0]
                break
        users.append(persona_data)

    # Create coordinate lists for each split using the original dataframes
    train_coords = [
        (
            row["persona_index"],
            prompt_to_idx[(row["prompt"], row["response_1"], row["response_2"])],
        )
        for _, row in df_train.iterrows()
    ]

    val_coords = [
        (
            row["persona_index"],
            prompt_to_idx[(row["prompt"], row["response_1"], row["response_2"])],
        )
        for _, row in df_validation.iterrows()
    ]

    calibration_coords = [
        (
            row["persona_index"],
            prompt_to_idx[(row["prompt"], row["response_1"], row["response_2"])],
        )
        for _, row in df_calibration.iterrows()
    ]

    test_coords = [
        (
            row["persona_index"],
            prompt_to_idx[(row["prompt"], row["response_1"], row["response_2"])],
        )
        for _, row in df_test.iterrows()
    ]

    n_train_pairs = df_train.shape[0] // 50
    n_train_users = len(df_train["persona_index"].drop_duplicates())

    return (
        preference_observations,
        pairs,
        instructions,
        train_coords,
        calibration_coords,
        val_coords,
        test_coords,
        n_train_users,
        n_train_pairs,
        users,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train preference learning model")

    # Add arguments
    parser.add_argument("--n_pairs", type=int, default=10000, help="Number of pairs")
    parser.add_argument("--n_users", type=int, default=1057, help="Number of users")
    parser.add_argument("--train_frac", type=float, default=0.95)
    parser.add_argument("--feature_dim", type=int, default=5, help="Feature dimension")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Number of gradient accumulation steps")
    parser.add_argument("--max_iterations", type=int, default=500, help="Maximum number of iterations")
    parser.add_argument("--name", type=str, default="debug4", help="Experiment name")
    parser.add_argument("--svd_init", type=bool, default=True, help="Use SVD initialization")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--regularization", type=str, default="l2", help="Regularization type, can be None, l1, l2, diag")
    parser.add_argument("--regularization_strength", type=float, default=0.02, help="Regularization strength")
    parser.add_argument("--save_model", type=bool, default=True, help="Save model")
    parser.add_argument("--dataset", type=str, default="PRISM", help="Dataset to use, options are PRISM")
    parser.add_argument( "--noise", type=float, default=0.0, help="Noise to add to the dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--user_type", type=str, default="curated", choices=["reduced", "filtered", "curated"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        n_pairs=args.n_pairs,
        n_users=args.n_users,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_iterations=args.max_iterations,
        name=args.name,
        svd_init=args.svd_init,
        random_seed=args.random_seed,
        regularization=args.regularization,
        regularization_strength=args.regularization_strength,
        save_model=args.save_model,
        noise=args.noise,
        model_name=args.model_name,
        train_frac=args.train_frac,
    )
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(0)
    random.seed(0)

    if args.dataset == "PRISM":
        train_data_path = "data/PRISM/preferences_dataset_train.csv"
        validation_data_path = "data/PRISM/preferences_dataset_validation.csv"
        calibration_data_path = "data/PRISM/preferences_dataset_calibration.csv"
        test_data_path = "data/PRISM/preferences_dataset_test.csv"

        df_train = pd.read_csv(train_data_path)
        df_validation = pd.read_csv(validation_data_path)
        df_calibration = pd.read_csv(calibration_data_path)
        df_test = pd.read_csv(test_data_path)
        (
            preference_observations,
            pairs,
            instructions,
            train_coords,
            calibration_coords,
            val_coords,
            test_coords,
            n_train_users,
            n_train_pairs,
            users,
        ) = prepare_prism_data(df_train, df_validation, df_calibration, df_test, config)
        config.n_users = preference_observations.shape[1]
        config.n_pairs = preference_observations.shape[0]
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    # create output directory
    output_dir = os.path.join("output", str(config))
    os.makedirs(output_dir, exist_ok=True)

    # save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f)

    # But the rest should be random based on the seed
    torch.manual_seed(config.random_seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    model = Model(
        config.n_pairs,
        config.n_users,
        config.feature_dim,
        normalize=False,
        bias=False,
        model_name=config.model_name,
    ).cuda()
    model.device = model.model.device
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    _, pairs1_batch, pairs2_batch, _, _, U_features = get_batch(
        config,
        preference_observations,
        None,
        pairs,
        instructions,
        train_coords,
    )
    # optionally use SVD to start.
    if config.svd_init:
        preference_observations_copy = preference_observations.clone()
        preference_observations_copy = preference_observations_copy[
            :n_train_pairs, :n_train_users
        ]
        if args.dataset == "PRISM":
            U, S, Vh = SVD_for_sparse_matrix(preference_observations_copy)
            # convert to tensors
            U = torch.tensor(U)
            S = torch.tensor(S)
            Vh = torch.tensor(Vh)
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")

        # init user vectors with V
        users_data = Vh.transpose(1, 0)[:, : config.feature_dim] * torch.sqrt(
            S[: config.feature_dim]
        ).unsqueeze(0)
        # Transform the rest of the users,
        heldout_users_data = torch.zeros(
            config.n_users - n_train_users, config.feature_dim, device=users_data.device
        )
        users_data = torch.concatenate([users_data, heldout_users_data], dim=0)
        model.set_users(users_data)
        model.users = model.users.to(model.device)
        # fit phi to U
        U = U[:, : config.feature_dim] * torch.sqrt(S[: config.feature_dim]).unsqueeze(0)
        # here pad with dummy data (won't ever be used)
        U = torch.concatenate(
            [
                U.cuda(),
                torch.zeros(config.n_pairs - n_train_pairs, config.feature_dim).cuda(),
            ],
            dim=0,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    model.linear_head = accelerator.prepare(model.linear_head)

    # SVD train
    loss_curve_svd = []
    if config.svd_init:
        loss_curve_batch = []

        for i in tqdm(
            range(config.svd_fit_iterations * config.gradient_accumulation_steps)
        ):
            with accelerator.accumulate(model):
                _, pairs1_batch, pairs2_batch, _, _, U_features = get_batch(
                    config,
                    preference_observations,
                    U,
                    pairs,
                    instructions,
                    train_coords,
                )
                pred_features = model._forward(pairs1_batch) - model._forward(
                    pairs2_batch
                )
                gt_features = U_features.to(torch.bfloat16)
                loss = torch.nn.functional.mse_loss(pred_features, gt_features)
                train_loss = loss.mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                loss_curve_batch.append(
                    train_loss.to(torch.float32).detach().cpu().numpy()
                )
                if i != 0 and i % config.gradient_accumulation_steps == 0:
                    loss_curve_svd.append(np.mean(loss_curve_batch))
                    loss_curve_batch = []

    # ROC of validation data after SVD train
    if config.svd_init:
        preferences_batchs = []
        predicted_probs = []
        for i in tqdm(range(len(val_coords) // config.batch_size)):
            (
                preferences_batch,
                pairs1_batch,
                pairs2_batch,
                users_indices,
                pairs_indices,
                _,
            ) = get_batch(
                config, preference_observations, None, pairs, instructions, val_coords
            )
            with torch.no_grad():
                preferences_batchs.append(preferences_batch)
                t = model.forward(pairs1_batch, pairs2_batch)
                indices = torch.tensor(users_indices).unsqueeze(1).cuda()
                pred_preferences = torch.gather(t, 1, indices)
                predicted_probs.append(torch.nn.functional.sigmoid(pred_preferences))
        predicted_probs = torch.stack(predicted_probs).flatten()
        preferences_batchs = torch.stack(preferences_batchs).flatten()

        predicted_probs = predicted_probs.to(torch.float32).cpu().detach().numpy()
        gt_labels = preferences_batchs.cpu().detach().numpy()
        svd_roc_auc, svd_max_accuracy = plot_roc(predicted_probs, gt_labels)
        print(f"ROC AUC: {svd_roc_auc:.2f}")
        print(f"Max Accuracy: {svd_max_accuracy:.2f}")
    else:
        svd_roc_auc = -1
        svd_max_accuracy = -1

    # MLE train
    for g in optimizer.param_groups:
        g["lr"] = 0.001

    loss_curve_mle_train = []
    loss_curve_batch = []
    loss_curve_mle_test = []
    for i in tqdm(range(config.max_iterations * config.gradient_accumulation_steps)):
        with accelerator.accumulate(model):
            (
                preferences_batch,
                pairs1_batch,
                pairs2_batch,
                users_indices,
                pairs_indices,
                _,
            ) = get_batch(
                config,
                preference_observations,
                None,
                pairs,
                instructions,
                train_coords,
                filter_halves=True,
            )
            if pairs1_batch == []:
                continue
            train_loss = calculate_loss_MLE(
                model,
                preferences_batch,
                pairs1_batch,
                pairs2_batch,
                users_indices,
                regularization=config.regularization,
                regularization_strength=config.regularization_strength,
            )
            accelerator.backward(train_loss)
            optimizer.step()
            optimizer.zero_grad()
            loss_curve_batch.append(train_loss.detach().cpu().to(torch.float32).numpy())
            if i != 0 and i % config.gradient_accumulation_steps == 0:
                loss_curve_mle_train.append(np.mean(loss_curve_batch))
                loss_curve_batch = []
            if i != 0 and i % (config.gradient_accumulation_steps * 10) == 0:
                with torch.no_grad():
                    test_losses = []
                    for _ in range(20):
                        (
                            preferences_batch,
                            pairs1_batch,
                            pairs2_batch,
                            users_indices,
                            pairs_indices,
                            _,
                        ) = get_batch(
                            config,
                            preference_observations,
                            None,
                            pairs,
                            instructions,
                            val_coords,
                            filter_halves=True,
                        )
                        if pairs1_batch == []:
                            continue
                        test_loss = calculate_loss_MLE(
                            model,
                            preferences_batch,
                            pairs1_batch,
                            pairs2_batch,
                            users_indices,
                        )
                        test_losses.append(test_loss)
                    test_loss = torch.stack(test_losses).mean()
                    loss_curve_mle_test.append(
                        test_loss.detach().cpu().to(torch.float32).numpy()
                    )

    # ROC of val after MLE training
    preferences_batchs = []
    predicted_probs = []
    for i in tqdm(range(len(val_coords) // config.batch_size)):
        (
            preferences_batch,
            pairs1_batch,
            pairs2_batch,
            users_indices,
            pairs_indices,
            _,
        ) = get_batch(
            config, preference_observations, None, pairs, instructions, val_coords
        )
        with torch.no_grad():
            preferences_batchs.append(preferences_batch)
            t = model.forward(pairs1_batch, pairs2_batch)
            indices = torch.tensor(users_indices).unsqueeze(1).cuda()
            pred_preferences = torch.gather(t, 1, indices)
            predicted_probs.append(torch.nn.functional.sigmoid(pred_preferences))
    predicted_probs = (
        torch.stack(predicted_probs).flatten().cpu().detach().to(torch.float32).numpy()
    )
    preferences_batchs = (
        torch.stack(preferences_batchs).flatten().cpu().detach().numpy()
    )
    mle_roc_auc, mle_max_accuracy = plot_roc(predicted_probs, preferences_batchs)
    print(f"Val ROC AUC: {mle_roc_auc:.2f}")
    print(f"Val Max Accuracy: {mle_max_accuracy:.2f}")

    def MLE_user_weight(embeddings, preferences, num_epochs=1000):
        H = embeddings.float()
        y = preferences.float()

        d = H.shape[1]
        u = torch.randn(d, requires_grad=True, device=H.device)

        optimizer = torch.optim.SGD([u], lr=0.01)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            logits = H @ u  # Shape: (M,)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

            loss.backward()
            optimizer.step()

        optimized_u = u.detach()
        return optimized_u, loss.cpu().detach().numpy()

    # evaluation on new users
    roc_auc_list = []
    max_accuracy_list = []
    for user in range(n_train_users, config.n_users):
        # extract user embedding from calibration data
        user_coords = [coord for coord in calibration_coords if coord[0] == user]
        preferences_batch, pairs1_batch, pairs2_batch, _, _, _ = get_batch(
            config,
            preference_observations,
            None,
            pairs,
            instructions,
            user_coords,
            batch_size=len(user_coords),
            filter_halves=True,
        )
        with torch.no_grad():
            embeddings_batch_list = []
            for i in range(0, len(pairs1_batch), config.batch_size):
                batch_end = min(i + config.batch_size, len(pairs1_batch))
                pairs1 = model._forward(pairs1_batch[i:batch_end])
                pairs2 = model._forward(pairs2_batch[i:batch_end])
                if model.normalize:
                    pairs1 = pairs1 / torch.linalg.norm(pairs1, axis=-1, keepdim=True)
                    pairs2 = pairs2 / torch.linalg.norm(pairs2, axis=-1, keepdim=True)
                embeddings_batch_list.append(pairs1 - pairs2)
            embeddings_batch = torch.cat(embeddings_batch_list, dim=0)
        user_embedding, _ = MLE_user_weight(embeddings_batch, preferences_batch)
        # evluate user on test data
        user_test_coords = [coord for coord in test_coords if coord[0] == user]
        preferences_batch, pairs1_batch, pairs2_batch, _, _, _ = get_batch(
            config,
            preference_observations,
            None,
            pairs,
            instructions,
            user_test_coords,
            batch_size=len(user_test_coords),
        )
        with torch.no_grad():
            embeddings_batch_list = []
            for i in range(0, len(pairs1_batch), config.batch_size):
                batch_end = min(i + config.batch_size, len(pairs1_batch))
                pairs1 = model._forward(pairs1_batch[i:batch_end])
                pairs2 = model._forward(pairs2_batch[i:batch_end])
                if model.normalize:
                    pairs1, pairs2 = (
                        pairs1 / torch.linalg.norm(pairs1, axis=-1, keepdim=True),
                        pairs2 / torch.linalg.norm(pairs2, axis=-1, keepdim=True),
                    )
                embeddings_batch_list.append(pairs1 - pairs2)
            embeddings_batch = torch.cat(embeddings_batch_list, dim=0)
            t = torch.matmul(embeddings_batch.float(), user_embedding.t())
            pred_preferences = torch.nn.functional.sigmoid(t)
        roc_auc, max_accuracy = plot_roc(
            pred_preferences.cpu().detach().numpy(),
            preferences_batch.cpu().detach().numpy(),
        )
        roc_auc_list.append(roc_auc)
        max_accuracy_list.append(max_accuracy)
    # print averages while ignoring nans
    print(f"Test ROC AUC: {np.nanmean(roc_auc_list):.2f}")
    print(f"Test Max Accuracy: {np.nanmean(max_accuracy_list):.2f}")

    results = {
        "svd_roc_auc": svd_roc_auc,
        "svd_max_accuracy": svd_max_accuracy,
        "svd_loss_curve": np.array(loss_curve_svd).flatten().tolist(),
        "mle_roc_auc": mle_roc_auc,
        "mle_max_accuracy": mle_max_accuracy,
        "mle_loss_curve": np.array(loss_curve_mle_train).flatten().tolist(),
        "mle_test_loss_curve": np.array(loss_curve_mle_test).flatten().tolist(),
        "new_users_roc_auc": np.nanmean(roc_auc_list),
        "new_users_max_accuracy": np.nanmean(max_accuracy_list),
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    # save model
    if config.save_model:
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
        test_users = list(range(n_train_users, config.n_users))
        with open(os.path.join(output_dir, "test_users.json"), "w") as f:
            json.dump(test_users, f)


if __name__ == "__main__":
    main()
