import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import random


def get_batch(
    cfg,
    preference_observations_mat,
    U_mat,
    pairs_tuple,
    instructions,
    coords,
    batch_size=None,
    filter_halves=False,
):
    if batch_size is None:
        batch_size = cfg.batch_size
    batch_coordinates = random.sample(coords, batch_size)
    preferences_batch = []
    pairs1_batch = []
    pairs2_batch = []
    U_features = []
    for users_idx, pairs_idx in batch_coordinates:
        preferences_batch.append(preference_observations_mat[pairs_idx, users_idx])
        instruction = instructions[pairs_idx]
        pairs1_batch.append(instruction + str(pairs_tuple[pairs_idx][0]))
        pairs2_batch.append(instruction + str(pairs_tuple[pairs_idx][1]))
        if U_mat is not None:
            U_features.append(U_mat[pairs_idx])
    if U_mat is not None:
        U_features = torch.stack(U_features).cuda().float()
    preferences_batch = torch.tensor(preferences_batch).cuda().float()
    users_indices = [item[0] for item in batch_coordinates]
    pairs_indices = [item[1] for item in batch_coordinates]
    if filter_halves:
        pairs1_batch = [
            pairs1_batch[i]
            for i in range(len(pairs1_batch))
            if preferences_batch[i] != 0.5
        ]
        pairs2_batch = [
            pairs2_batch[i]
            for i in range(len(pairs2_batch))
            if preferences_batch[i] != 0.5
        ]
        users_indices = [
            users_indices[i]
            for i in range(len(users_indices))
            if preferences_batch[i] != 0.5
        ]
        pairs_indices = [
            pairs_indices[i]
            for i in range(len(pairs_indices))
            if preferences_batch[i] != 0.5
        ]
        if U_mat is not None:
            U_features = U_features[preferences_batch != 0.5]
        preferences_batch = preferences_batch[preferences_batch != 0.5]

    return (
        preferences_batch,
        pairs1_batch,
        pairs2_batch,
        users_indices,
        pairs_indices,
        U_features,
    )


def calculate_loss_MLE(
    model,
    preferences_batch,
    pairs1_batch,
    pairs2_batch,
    users_indices,
    individual_user=False,
    regularization=None,
    regularization_strength=None,
):
    """Calculate MLE loss for preference learning.
    For individual user training, we don't need to gather specific user predictions.
    """
    # Check if we have any valid samples after filtering
    if len(pairs1_batch) == 0 or len(pairs2_batch) == 0:
        return torch.tensor(0.0, requires_grad=True, device="cuda")

    t, features = model.forward(pairs1_batch, pairs2_batch, return_features=True)

    if individual_user:
        # For individual user model, just use the predictions directly
        pred_preferences = t.squeeze(1)  # Remove the user dimension
    else:
        # For multi-user model, gather the specific user predictions
        # Ensure indices are within bounds
        max_idx = t.size(1) - 1
        indices = torch.tensor(
            [min(idx, max_idx) for idx in users_indices], device="cuda"
        ).unsqueeze(1)
        pred_preferences = torch.gather(t, 1, indices).squeeze(1)

    # Ensure preferences_batch and pred_preferences have the same shape
    if pred_preferences.shape != preferences_batch.shape:
        raise ValueError(
            "pred_preferences and preferences_batch have different shapes. shapes: ",
            pred_preferences.shape,
            preferences_batch.shape,
        )

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_preferences, preferences_batch, reduction="none"
    )
    if regularization is not None:
        if regularization == "l1":
            loss = (
                loss + regularization_strength * torch.norm(features, 1, dim=1).mean()
            )
        elif regularization == "l2":
            loss = (
                loss + regularization_strength * torch.norm(features, 2, dim=1).mean()
            )
        elif regularization == "double_l2":
            reg_loss_features = torch.norm(features, 2, dim=1).mean()
            reg_loss_users = torch.norm(model.users, 2, dim=1).mean()
            loss = loss + regularization_strength * (reg_loss_features + reg_loss_users)
        elif regularization == "diag":
            B = features.T @ features
            B = torch.clamp(B, min=-1e6, max=1e6)
            with torch.no_grad():
                diagonal_mask = 1 - torch.eye(B.size(0), device=B.device)
            penalty_rotation = torch.sum(torch.abs((B * diagonal_mask)))
            loss = loss + regularization_strength * penalty_rotation
        elif regularization == "diag_full":
            B = features.T @ features
            B = torch.clamp(B, min=-1e6, max=1e6)
            with torch.no_grad():
                diagonal_mask = 1 - torch.eye(B.size(0), device=B.device)
            penalty_rotation = torch.sum(torch.abs((B * diagonal_mask)))
            penalty_scale = torch.sum(torch.abs(torch.diag(B) - 1.0))
            loss = loss + regularization_strength * (penalty_rotation + penalty_scale)
    return loss.mean()


def plot_roc(predicted_probs, gt_labels, plot=False):
    valid_indices = gt_labels != 0.5
    filtered_gt_labels = gt_labels[valid_indices]
    filtered_pred_probs = predicted_probs[valid_indices]

    # Compute the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(filtered_gt_labels, filtered_pred_probs)
    roc_auc = auc(fpr, tpr)

    # find the maximum accuracy
    max_accuracy = 0
    max_threshold = 0
    for threshold in thresholds:
        binarized_pred = np.array(filtered_pred_probs) > threshold
        accuracy = np.mean(filtered_gt_labels == binarized_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_threshold = threshold

    # Plot the ROC curve
    if plot:
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc, max_accuracy


def to_preference(text):
    if text == "('Output (a)', 'Output (b)')":
        return 1
    elif text == "('Output (b)', 'Output (a)')":
        return 0
    elif text == "('Output (a)', 'Output (a)')":
        return 0.5
    elif text == "('Output (b)', 'Output (b)')":
        return 0.5
    else:
        return 0.5


def plot_loss_curves(loss_curve, output_dir, save_to_disk=False):
    window = 5
    average_y = []
    for ind in range(len(loss_curve) - window + 1):
        average_y.append(np.mean(loss_curve[ind : ind + window]))

    plt.figure()
    plt.plot(average_y, label="loss")
    plt.title("MSE loss")
    plt.legend()
    if save_to_disk:
        plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()


def SVD_for_sparse_matrix(preference_observations_mat, N=10, feature_dim=100):
    """
    based on https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/missing-value-Kurucz.pdf
    1. impute missing values with 0.5
    2. compute SVD
    3. get the low rank approximation of the matrix
    4. impute the missing values with the SVD approximation
    5. repeat N times
    """
    # Convert input to numpy array if not already
    preference_observations = np.array(preference_observations_mat, dtype=np.float32)

    # Get mask of observed (non-missing) values
    observed_mask = ~np.isnan(preference_observations)

    # Initialize missing values with 0.5
    preference_observations[~observed_mask] = 0.5

    # Iteratively improve SVD approximation
    for i in range(N):
        # Compute SVD
        U, S, V = np.linalg.svd(preference_observations, full_matrices=False)

        # Keep only top k singular values/vectors for low rank approximation
        k = min(feature_dim, len(S))  # Use 8 or fewer components
        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V[:k, :]

        # Get low rank approximation using truncated matrices
        low_rank = (U_k * S_k) @ V_k

        # Update only missing values with SVD approximation
        preference_observations[~observed_mask] = low_rank[~observed_mask]

        # Clamp values to valid preference range [0,1]
        preference_observations = np.clip(preference_observations, 0, 1)

    return U, S, V


def get_batch_vlp(
    cfg,
    preference_observations_mat,
    pairs_tuple,
    instructions,
    coords,
    K=5,
    batch_size=None,
    filter_halves=False,
):
    """Get a batch of data with additional context items for each user.

    Args:
        cfg: Config object
        preference_observations_mat: Matrix of preferences
        pairs_tuple: Tuple of pairs
        instructions: List of instructions
        coords: List of coordinates
        K: Number of context items per user
        batch_size: Batch size (if None, use cfg.batch_size)
        filter_halves: Whether to filter out 0.5 preferences

    Returns:
        Tuple containing batch data and context data
    """
    if batch_size is None:
        batch_size = cfg.batch_size

    # Get regular batch first
    preferences_batch, pairs1_batch, pairs2_batch, users_indices, pairs_indices, _ = (
        get_batch(
            cfg,
            preference_observations_mat,
            None,
            pairs_tuple,
            instructions,
            coords,
            batch_size=batch_size,
            filter_halves=filter_halves,
        )
    )
    # all coords are in the form (user_idx, pair_idx)

    if len(preferences_batch) == 0:
        return (
            preferences_batch,
            pairs1_batch,
            pairs2_batch,
            users_indices,
            pairs_indices,
            None,
            None,
            None,
        )

    # Get context items for each user
    context_preferences = []
    context_pairs1 = []
    context_pairs2 = []

    for i, user_idx in enumerate(users_indices):
        # Get all pairs for this user
        user_pairs = [(k, j) for k, j in coords if k == user_idx]

        # Remove the pairs that are in the main batch
        user_pairs = [p for p in user_pairs if p[1] != pairs_indices[i]]

        # Initialize context lists for this user
        user_context_preferences = []
        user_context_pairs1 = []
        user_context_pairs2 = []

        # Sample K pairs (or all if less than K available)
        K_actual = min(K, len(user_pairs))
        if K_actual > 0:
            context_coordinates = random.sample(user_pairs, K_actual)

            # Get data for context pairs
            for users_idx, pairs_idx in context_coordinates:
                user_context_preferences.append(
                    preference_observations_mat[pairs_idx, users_idx]
                )
                instruction = instructions[pairs_idx]
                user_context_pairs1.append(instruction + pairs_tuple[pairs_idx][0])
                user_context_pairs2.append(instruction + pairs_tuple[pairs_idx][1])

        # Add this user's context to the main lists
        context_preferences.append(user_context_preferences)
        context_pairs1.append(user_context_pairs1)
        context_pairs2.append(user_context_pairs2)

    # Convert preferences to 2D tensor, padding with zeros if needed
    max_context_len = max(len(prefs) for prefs in context_preferences)
    if max_context_len == 0:
        return [], [], [], [], [], None, None, None

    padded_preferences = (
        torch.zeros(len(context_preferences), max_context_len).cuda().float()
    )
    for i, prefs in enumerate(context_preferences):
        if len(prefs) > 0:
            padded_preferences[i, : len(prefs)] = torch.tensor(prefs).float()

    return (
        preferences_batch,
        pairs1_batch,
        pairs2_batch,
        users_indices,
        pairs_indices,
        padded_preferences,
        context_pairs1,
        context_pairs2,
    )
