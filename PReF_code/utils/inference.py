import numpy as np
import torch
import os
from tqdm import tqdm
import pandas as pd

from pairs.completions import get_completion
import pairs.prompts as prompts

def sigmoid(x):
  """
  Sigmoid function that supports numpy vectors.
  """
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function that supports numpy vectors.
    """
    s = sigmoid(x)
    return s * (1 - s)


def get_embeddings(df, dfs_responses, base_responses, model_dir, model, num_test_pairs=100):
    num_pairs = len(df)

    if not os.path.exists(os.path.join(model_dir, "test_embeddings.pth")):
        test_embeddings = []
        for i in tqdm(range(num_pairs-num_test_pairs, num_pairs)):
            with torch.no_grad():
                vec_1 = model._forward(df['instruction'][i] + df['response_1'][i])
                vec_2 = model._forward(df['instruction'][i] + df['response_2'][i])
                embedding = vec_1-vec_2
            test_embeddings.append(embedding)
        test_embeddings = torch.stack(test_embeddings).squeeze().cpu()
        torch.save(test_embeddings, os.path.join(model_dir, "test_embeddings.pth"))
    else:
        test_embeddings = torch.load(os.path.join(model_dir, "test_embeddings.pth"))

    if not os.path.exists(os.path.join(model_dir, "train_embeddings.pth")):
        train_embeddings = []
        for i in tqdm(range(num_pairs-num_test_pairs)):
            with torch.no_grad():
                vec_1 = model._forward(df['instruction'][i] + df['response_1'][i])
                vec_2 = model._forward(df['instruction'][i] + df['response_2'][i])
                embedding = vec_1-vec_2
            train_embeddings.append(embedding)
        train_embeddings = torch.stack(train_embeddings).squeeze()
        torch.save(train_embeddings, os.path.join(model_dir, "train_embeddings.pth"))
    else:
        train_embeddings = torch.load(os.path.join(model_dir, "train_embeddings.pth"))

    if not os.path.exists(os.path.join(model_dir, "dfs_embeddings.pth")):
        dfs_embeddings = []
        for i in tqdm(range(len(dfs_responses))):
            df = dfs_responses[i]
            embeddings = []
            for j in range(len(df)):
                with torch.no_grad():
                    vec = model._forward(df['instruction'][j] + df['responses'][j])
                    embeddings.append(vec)
            embeddings = torch.stack(embeddings).squeeze().cpu()
            dfs_embeddings.append(embeddings)
        dfs_embeddings = torch.stack(dfs_embeddings)
        torch.save(dfs_embeddings, os.path.join(model_dir, "dfs_embeddings.pth"))
    else:
        dfs_embeddings = torch.load(os.path.join(model_dir, "dfs_embeddings.pth"))
    
    return test_embeddings, train_embeddings, dfs_embeddings


def MLE_user_weight(item_embeddings, item_preferences, user_initial=None, num_epochs = 5000):
    H = torch.tensor(item_embeddings)
    y = torch.tensor(item_preferences).float().to(H.device)

    d = H.shape[1]
    if user_initial is None:
        u = torch.randn(d, requires_grad=True, device=H.device)
    else:
        u = torch.tensor(user_initial, requires_grad=True, device=H.device)
    
    optimizer = torch.optim.SGD([u], lr=0.01)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        logits = H @ u  # Shape: (M,)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        
        loss.backward()
        optimizer.step()
        
    optimized_u = u.detach()
    return optimized_u, loss.cpu().detach().numpy()

def choose_personalized_response(user_embedding, responses_embeddings, responses_dataframe):
    responses_embeddings = responses_embeddings

    max_value = -np.inf
    max_index = None
    
    for i,v in enumerate(responses_embeddings):
        value = v.T @ user_embedding
        if value > max_value:
            max_value = value
            max_index = i

    return responses_dataframe['responses'][max_index]

def max_norm_vector(candidates, V, M, u, use_derivitaive=False):
    """
    Finds the vector in V that maximizes the expression Sigma'(u^v) (v)^T M^-1 (v)

    Parameters:
    V (numpy.ndarray): A 2D array where each row is a vector.
    M (numpy.ndarray): A square matrix of appropriate dimensions.
    u (numpy.ndarray): A vector
    """
    # Compute the inverse of matrix M
    M_inv = np.linalg.inv(M)

    # Initialize variables to keep track of the maximum value and corresponding vector
    max_value = -np.inf
    max_index = None

    # Iterate over each vector in V
    for i in candidates:
        # Compute the value of (v)^T M^-1 (v)
        v = V[i,:]
        value = v.T @ M_inv @ v
        if use_derivitaive:
            coeff = sigmoid_derivative(u.T @ v)
            value = coeff**2 * value

        # Update the maximum value and corresponding vector if needed
        if value > max_value:
            max_value = value
            max_index = i
    
    return max_index