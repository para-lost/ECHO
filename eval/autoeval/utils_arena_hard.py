import glob
import pandas as pd
import torch
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from dataclasses import dataclass
from functools import partial
from typing import Dict, Callable, Optional

REGISTER_LOSSES: Dict[str, Callable] = {}
REGISTER_MODELS: Dict[str, Callable] = {}

def get_score(judgment, patterns):
    import re
    j = re.sub(r"\s+", "", judgment.upper()) 
    for pattern in patterns:
        pattern = re.compile(pattern)
        matches = pattern.findall(j)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None

# ======================
#     math_utils.py
# ======================

def register_loss(name: str):
    def decorator(func: Callable):
        REGISTER_LOSSES[name] = func
        return func

    return decorator


def register_model(name: str):
    def decorator(func: Callable):
        REGISTER_MODELS[name] = func
        return func

    return decorator


@dataclass
class ModelParams:
    coefs: torch.FloatTensor = None
    eta: Optional[torch.FloatTensor] = None


@register_model("bt")
class BTModel(nn.Module):
    def __init__(self, num_components):
        super().__init__()
        self.logits = nn.Parameter(
            nn.init.constant_(torch.empty(num_components), 0.5)
        )

    def forward(self):
        return self.logits, None


@register_model("rk")
class RKModel(nn.Module):
    def __init__(self, num_components):
        super().__init__()
        self.logits = nn.Parameter(
            nn.init.constant_(torch.empty(num_components), 0.5)
        )
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self):
        return self.logits, self.eta
    

@register_loss("bt")
def bt_loss(
    logits: torch.Tensor,
    outcomes: torch.Tensor,
    alpha: float = 0.5,
    **kwargs,
):
    # more stable than the original implementation
    loss = F.binary_cross_entropy_with_logits(
        logits,
        outcomes.float(),
        reduction='sum'
    )
    
    return loss
    

@register_loss("rk")
def rk_loss(
    logits: torch.Tensor,
    outcomes: torch.Tensor,
    eta: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-10,
    **kwargs,
):    
    logits = torch.where(outcomes == 0, -logits, logits)
    
    probs_w = torch.sigmoid(logits - eta)
    probs_l = torch.sigmoid(-1 * logits - eta)
    probs_t = 1 - probs_w - probs_l

    # point-wise likelihood
    ties = (outcomes == 0.5).long() # TODO: Ties must be 0.5
 
    A = torch.stack((probs_w, probs_t))
    p = A.take_along_dim(dim=0, indices=ties.unsqueeze(0))

    loss = -torch.log(p + eps).mean()
    
    return loss


def fit_pairwise_model(
    features: torch.Tensor, 
    outcomes: torch.Tensor,
    loss_type: str = 'bt', 
    indices: torch.Tensor = None,
    lr: float = 0.1, 
    tol: float = 1e-9, 
    max_epochs: int = 50
):
    model_cls = REGISTER_MODELS[loss_type]
    loss_func = REGISTER_LOSSES[loss_type]
    
    if indices is not None:
        features = features[indices]
        outcomes = outcomes[indices]
        
    assert not features.isnan().any()
    
    model = model_cls(num_components=features.shape[1])

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_epochs,
        tolerance_grad=tol,
        tolerance_change=tol,
    )

    def closure():
        optimizer.zero_grad()
        logits, eta = model()
                
        _logits = features @ logits
        
        loss = loss_func(
            logits=_logits, 
            outcomes=outcomes, 
            eta=eta
        )
        loss.backward()
        return loss

    optimizer.step(closure)

    logits, eta = model()
    return logits.detach(), eta if eta is None else eta.detach()


def worker_fn_pairwise_model(features, outcomes, loss_type, boot_idxs, idx):
    indices = boot_idxs[idx]
    return fit_pairwise_model(features, outcomes, loss_type, indices)


def bootstrap_pairwise_model(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    loss_type: str = "bt",
    num_round: int = 100,
):
    boot_idxs = np.random.randint(
        low=0, high=features.shape[0], 
        size=(num_round, features.shape[0])
    )
    
    results = [fit_pairwise_model(features, outcomes, loss_type, boot_idxs[i]) for i in tqdm(range(num_round))]
        
    logit_stacks = torch.stack([result[0] for result in results])
    
    if results[0][1] is not None:
        eta_stacks = torch.stack([result[1] for result in results])
    else:
        eta_stacks = None
    
    return logit_stacks, eta_stacks


def fit_binary_model(
    features: np.ndarray,
    outcomes: np.ndarray,
    indices: np.ndarray = None,
    max_iter: int = 1000,
):
    from sklearn.linear_model import LogisticRegression
    
    if indices is not None:
        features = features[indices]
        outcomes = outcomes[indices]
    
    model = LogisticRegression(max_iter=max_iter)
    model.fit(features, outcomes)
    
    return model.coef_, model.intercept_


def worker_fn_binary_model(idx, features, outcomes, boot_idxs):
    indices = boot_idxs[idx]
    return fit_binary_model(features, outcomes, indices)


def bootstrap_binary_model(
    features: np.ndarray,
    outcomes: np.ndarray,
    num_round: int = 100,
    num_cpu: Optional[int] = None,
):
    
    boot_idxs = np.random.randint(
        low=0, high=features.shape[0], 
        size=(num_round, features.shape[0])
    )
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    worker = partial(
        worker_fn_binary_model,
        features,
        outcomes, 
        boot_idxs
    )
    
    num_cpu = num_cpu if num_cpu else os.cpu_count() // 4
    print(f"INFO: Using {num_cpu} CPUs for bootstrapping.")
    
    with mp.Pool(num_cpu) as pool:
        results = list(
            tqdm(pool.imap(worker, range(num_round)), total=num_round)
        )

    coef_stacks = [result[0] for result in results]
    intercept_stacks = [result[1] for result in results]
    
    return coef_stacks, intercept_stacks


def one_hot_encode(items, baseline="o3-mini-2025-01-31"):
    # Get unique items and sort them
    unique_items = sorted(set(items + [baseline]))
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}

    # Initialize the one-hot encoded matrix
    one_hot_matrix = []

    for item in items:
        # Create a zero-filled list
        one_hot_vector = [0] * len(unique_items)
        # Set the appropriate index to 1
        one_hot_vector[item_to_index[item]] = 1
        one_hot_vector[item_to_index[baseline]] = -1
        one_hot_matrix.append(one_hot_vector)

    return torch.tensor(one_hot_matrix, dtype=torch.float32), unique_items


def to_winrate_probabilities(
    coefs,
    models,
    baseline_model="o3-mini-2025-01-31",
):  
    baseline_idx = models.index(baseline_model)
    
    exp_coefs = torch.exp(coefs)
    probs = torch.zeros(coefs.shape[0], coefs.shape[1])
    
    for idx in range(len(models)):
        if models[idx] == baseline_model:
            probs[:, idx] = 0.5
        probs[:, idx] = exp_coefs[:, idx] / (exp_coefs[:, idx] + exp_coefs[:, baseline_idx])
        
    return probs

# ======================
#     show_result.py
# ======================

def load_judgments(judge_names, benchmark, weight=3):
    dfs = []
    for judge_name in judge_names:
        print(f"Loading {judge_name} judgments...")
        dfs.extend([
            pd.read_json(f, lines=True) for f in tqdm(glob(os.path.join(
                "data",
                benchmark, 
                "model_judgment", 
                judge_name, 
                "*.jsonl"
            )))
        ])
    data = pd.concat(dfs).reset_index(drop=True)
    
    # if data.model.isin(judge_names).any():
    #     print(f"WARNING: {judge_names} is already in the data. Removing it.")
    #     data = data[~data.model.isin(judge_names)].reset_index(drop=True)

    null_indices = data.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
    _data = data[~null_indices].reset_index(drop=True)
    
    print(f"Number of null judgments found: {len(data) - len(_data)}")
    
    # map label to score
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    _data['scores'] = _data.games.map(
        lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    )
    
    battles = _data[['uid', 'model', 'category', 'scores']].explode('scores').reset_index(drop=True)
    
    return battles


def format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline=None, mult=100):
    leaderboard = pd.merge(
        mean_scores, 
        lower_scores, 
        on="model"
    ).merge(
        upper_scores, 
        on="model"
    )
    
    leaderboard["Scores (%)"] = leaderboard["scores"].map(lambda x: round(x * mult, 1))
    
    leaderboard["CI (%)"] = leaderboard.apply(
        lambda row: f"(-{round((row['scores'] - row['lower']) * mult, 1)} / +{round((row['upper'] - row['scores']) * mult, 1)})", 
        axis=1
    )
    
    _leaderboard = leaderboard.rename(
        columns={"model": "Model"}
    ).drop(
        columns=["lower", "upper", "scores"]
    )
    
    if baseline:
        _leaderboard = pd.concat(
            [_leaderboard, pd.DataFrame({"Model": baseline, "Scores (%)": 50.0, "CI (%)": "(-0.0 / +0.0)"}, index=[0])]
        )
    
    return _leaderboard.sort_values(by="Scores (%)", ascending=False).reset_index(drop=True)


def print_leaderboard(battles, category, baseline=None, print_leaderboard=False):
    # baseline = JUDGE_SETTINGS[category]["baseline"]
    
    _battles = battles.drop(columns=['category'])[['model', 'scores']]
    
    # remove model path
    _battles['model'] = _battles['model'].map(lambda x: x.split('/')[-1])
    
    bootstraps = pd.concat([
        _battles.groupby("model").sample(frac=1.0, replace=True).groupby("model").mean()
        for _ in tqdm(range(100))
    ])
    
    bootstraps["scores"] = bootstraps["scores"].astype(float)
    
    mean_scores = bootstraps.groupby("model").mean().reset_index()
    lower_scores = bootstraps.groupby("model").quantile(0.05).reset_index().rename(columns={"scores": "lower"})
    upper_scores = bootstraps.groupby("model").quantile(0.95).reset_index().rename(columns={"scores": "upper"})
    
    _leaderboard = format_confidence_interval(mean_scores, lower_scores, upper_scores, baseline)
    
    if print_leaderboard:
        print(f"##### Category: {category} #####")
        print(_leaderboard.to_string())

    return _leaderboard