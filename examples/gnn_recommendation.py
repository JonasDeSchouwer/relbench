import argparse
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from collections import namedtuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from model import Model
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, Table, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import LinkNeighborLoader
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
# Use the same seed time across the mini-batch and share the negatives
parser.add_argument("--share_same_time", action="store_true", default=True)
parser.add_argument(
    "--no-share_same_time", dest="share_same_time", action="store_false"
)
# Whether to use shallow embedding on dst nodes or not.
parser.add_argument("--use_shallow", action="store_true", default=True)
parser.add_argument("--no-use_shallow", dest="use_shallow", action="store_false")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
assert task.task_type == TaskType.LINK_PREDICTION

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

train_table_input = get_link_train_table_input(task.get_table("train"), task)
train_loader = LinkNeighborLoader(
    data=data,
    num_neighbors=num_neighbors,
    time_attr="time",
    src_nodes=train_table_input.src_nodes,
    dst_nodes=train_table_input.dst_nodes,
    num_dst_nodes=train_table_input.num_dst_nodes,
    src_time=train_table_input.src_time,
    share_same_time=args.share_same_time,
    batch_size=args.batch_size,
    temporal_strategy=args.temporal_strategy,
    # if share_same_time is True, we use sampler, so shuffle must be set False
    shuffle=not args.share_same_time,
    num_workers=args.num_workers,
)

eval_loaders_dict: Dict[str, Tuple[NeighborLoader, NeighborLoader]] = {}
for split in ["val", "test"]:
    timestamp = dataset.val_timestamp if split == "val" else dataset.test_timestamp
    seed_time = int(timestamp.timestamp())
    target_table = task.get_table(split)
    src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)
    src_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=(task.src_entity_table, src_node_indices),
        input_time=torch.full(
            size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dst_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=task.dst_entity_table,
        input_time=torch.full(
            size=(task.num_dst_nodes,), fill_value=seed_time, dtype=torch.long
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    eval_loaders_dict[split] = (src_loader, dst_loader)

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=args.channels,
    aggr=args.aggr,
    norm="layer_norm",
    shallow_list=[task.dst_entity_table] if args.use_shallow else [],
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(train_loader), args.max_steps_per_epoch)
    for batch in tqdm(train_loader, total=total_steps):
        src_batch, batch_pos_dst, batch_neg_dst = batch
        src_batch, batch_pos_dst, batch_neg_dst = (
            src_batch.to(device),
            batch_pos_dst.to(device),
            batch_neg_dst.to(device),
        )
        x_src = model(src_batch, task.src_entity_table)
        x_pos_dst = model(batch_pos_dst, task.dst_entity_table)
        x_neg_dst = model(batch_neg_dst, task.dst_entity_table)

        # [batch_size, ]
        pos_score = torch.sum(x_src * x_pos_dst, dim=1)
        if args.share_same_time:
            # [batch_size, batch_size]
            neg_score = x_src @ x_neg_dst.t()
            # [batch_size, 1]
            pos_score = pos_score.view(-1, 1)
        else:
            # [batch_size, ]
            neg_score = torch.sum(x_src * x_neg_dst, dim=1)
        optimizer.zero_grad()
        # BPR loss
        diff_score = pos_score - neg_score
        loss = F.softplus(-diff_score).mean()
        loss.backward()
        optimizer.step()

        loss_accum += float(loss) * x_src.size(0)
        count_accum += x_src.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.compile
def get_scores_and_indices(emb: Tensor, dst_emb: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        emb: (batch_size, d)
        dst_emb: (num_dst_nodes, d)
    Returns:
        scores: (batch_size, num_dst_nodes)
        indices: (batch_size, num_dst_nodes)
    """
    return torch.sort(emb @ dst_emb.t(), dim=1, descending=True) # (batch_size, num_dst_nodes)


@torch.no_grad()
def test(src_loader: NeighborLoader, dst_loader: NeighborLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    dst_embs: list[Tensor] = []
    for batch in tqdm(dst_loader):
        batch = batch.to(device)
        emb = model(batch, task.dst_entity_table).detach()
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    pred_index_mat_list: list[Tensor] = []
    pred_scores_sorted_list: list[Tensor] = []
    for batch in tqdm(src_loader):
        batch = batch.to(device)
        emb = model(batch, task.src_entity_table) # (batch_size, d)

        # sorted indices of the dst predictions, descending order
        pred_scores_sorted, pred_index_mat = get_scores_and_indices(emb, dst_emb) # (batch_size, num_dst_nodes)
        pred_index_mat_list.append(pred_index_mat.cpu())
        pred_scores_sorted_list.append(pred_scores_sorted.cpu())
    pred = torch.cat(pred_index_mat_list, dim=0).numpy() # (num_src_nodes, num_dst_nodes)
    pred_scores_sorted = torch.cat(pred_scores_sorted_list, dim=0).numpy() # (num_src_nodes, num_dst_nodes)
    return pred, pred_scores_sorted


def plot_logit_distribution(pos_logits: np.ndarray, neg_logits: np.ndarray) -> plt.Figure:
    """
    Args:
        pos_logits: (num_pos_logits,)
        neg_logits: (num_neg_logits,)
    """
    fig, ax1 = plt.subplots()
    ax1.hist(pos_logits, bins=100, alpha=0.5, label="Positive", color="green")
    ax1.set_ylabel("Positive count")

    ax2 = ax1.twinx()
    ax2.hist(neg_logits, bins=100, alpha=0.5, label="Negative", color="red")
    ax2.set_ylabel("Negative count")

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    return fig


def plot_rankings(pred_isin: np.ndarray, eval_k: int):
    """
    Create scatter plot showing points where pred_isin is True
    Args: 
        pred_isin: (num_src_nodes, num_dst_nodes) - bool
    Returns:
        namedtuple("RankingPlots", ["fig_scatter", "ax_scatter", "fig_zoomed", "ax_zoomed"])
    """
    rows, cols = np.where(pred_isin)
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    ax_scatter.scatter(cols, rows, s=1, alpha=0.5)
    ax_scatter.set_xlim(-0.5, pred_isin.shape[-1] - 0.5)
    ax_scatter.set_xlabel("rank")
    ax_scatter.set_ylabel("src_idx")
    ax_scatter.set_title(f"Scatter plot of True values in pred_isin ({len(rows)} points)")
    ax_scatter.axvline(x=eval_k-0.5, color="red", linestyle="--", linewidth=1, label="rank=10")
    ax_scatter.legend()
    ax_scatter.invert_yaxis()  # Invert y-axis to match imshow orientation
    plt.tight_layout()

    # Create zoomed-in scatter plot focusing on first 100 ranks
    MAX_RANK = 100
    mask = cols < MAX_RANK
    rows_zoomed = rows[mask]
    cols_zoomed = cols[mask]

    fig_zoomed, ax_zoomed = plt.subplots(figsize=(12, 8))
    ax_zoomed.scatter(cols_zoomed, rows_zoomed, s=2, alpha=0.6)
    ax_zoomed.set_xlabel("rank")
    ax_zoomed.set_ylabel("src_idx")
    ax_zoomed.set_title(f"Zoomed: True values in first {MAX_RANK} ranks ({len(rows_zoomed)} points)")
    ax_zoomed.axvline(x=eval_k-0.5, color="red", linestyle="--", linewidth=1, label="rank=10")
    ax_zoomed.set_xlim(-0.5, MAX_RANK - 0.5)
    ax_zoomed.legend()
    ax_zoomed.invert_yaxis()  # Invert y-axis to match imshow orientation
    ax_zoomed.grid(True, alpha=0.3)
    plt.tight_layout()

    return namedtuple("RankingPlots", ["fig_scatter", "ax_scatter", "fig_zoomed", "ax_zoomed"])(fig_scatter, ax_scatter, fig_zoomed, ax_zoomed)


def get_pred_isin_and_dst_count(
        pred: NDArray,
        target_table: Table,
        dst_entity_col: str,
    ) -> Dict[str, float]:
        pred_isin_list = []
        dst_count_list = []
        for true_dst_nodes, pred_dst_nodes in tqdm(zip(
            target_table.df[dst_entity_col],
            pred,
        ), total=len(pred)):
            pred_isin_list.append(
                np.isin(np.array(pred_dst_nodes), np.array(true_dst_nodes))
            )
            dst_count_list.append(len(true_dst_nodes))
        pred_isin = np.stack(pred_isin_list)
        dst_count = np.array(dst_count_list)

        return pred_isin, dst_count


def auc(pred_isin: NDArray, dst_count: NDArray):
    """
    Args:
        pred_isin: (num_src_nodes, num_dst_nodes). dim -1 is ordered by descending score. We want to know the AUC score of the prediction.
        dst_count: (num_src_nodes,)

    Gets the AUC score of the prediction.
    """
    num_dst_nodes = pred_isin.shape[-1]
    # cum[i] is the number of 1s in [0, ..., i]
    cum = np.cumsum(pred_isin, axis=-1) # (num_src_nodes, num_dst_nodes)

    # remove the cases where dst_count is 0 or num_dst_nodes
    valid_mask = (dst_count > 0) & (dst_count < num_dst_nodes)
    auc = (1 / (dst_count * (num_dst_nodes - dst_count))[valid_mask]) * (cum * (pred_isin == 0)).sum(axis=-1)[valid_mask]

    return auc.mean()


# eval at epoch 0:
epoch = 0

val_pred, val_pred_scores_sorted = test(*eval_loaders_dict["val"])
val_pred_isin, val_dst_count = get_pred_isin_and_dst_count(val_pred, task.get_table("val"), task.dst_entity_col)
val_auc = auc(val_pred_isin, val_dst_count)
val_metrics = {
    "auc": val_auc,
}
print(
    f"Epoch: {epoch:02d} - Initial evaluation"
    f"Val metrics: {val_metrics}"
)
# plots
# ranking - first 2000
val_ranking_plots = plot_rankings(val_pred_isin[:2000], task.eval_k)
val_ranking_plots.fig_scatter.savefig(f"output/{args.dataset}_{args.task}/figures/2000_ranking_{epoch}.png")
val_ranking_plots.fig_zoomed.savefig(f"output/{args.dataset}_{args.task}/figures/2000_ranking_zoomed_{epoch}.png")
plt.close(val_ranking_plots.fig_scatter)
plt.close(val_ranking_plots.fig_zoomed)
print("Generated val ranking plot")

# logits distribution - first 5000
val_logit_distribution_plots = plot_logit_distribution(val_pred_scores_sorted[:5000][val_pred_isin[:5000]], val_pred_scores_sorted[:5000][~val_pred_isin[:5000]])
val_logit_distribution_plots.savefig(f"output/{args.dataset}_{args.task}/figures/5000_logit_distribution_{epoch}.png")
plt.close(val_logit_distribution_plots)
print("Generated val logit distribution plot")


# === our training loop ===
# who art in the cloud
state_dict = None
best_val_metric = 0
tune_metric = "auc"
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred, val_pred_scores_sorted = test(*eval_loaders_dict["val"])
        val_pred_isin, val_dst_count = get_pred_isin_and_dst_count(val_pred, task.get_table("val"), task.dst_entity_col)
        val_auc = auc(val_pred_isin, val_dst_count)
        val_metrics = {
            "auc": val_auc,
        }
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        if val_metrics[tune_metric] >= best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
        
        # plots
        # ranking - first 2000
        val_ranking_plots = plot_rankings(val_pred_isin[:2000], task.eval_k)
        val_ranking_plots.fig_scatter.savefig(f"output/{args.dataset}_{args.task}/figures/2000_ranking_{epoch}.png")
        val_ranking_plots.fig_zoomed.savefig(f"output/{args.dataset}_{args.task}/figures/2000_ranking_zoomed_{epoch}.png")
        plt.close(val_ranking_plots.fig_scatter)
        plt.close(val_ranking_plots.fig_zoomed)
        print("Generated val ranking plot")

        # logits distribution - first 5000
        val_logit_distribution_plots = plot_logit_distribution(val_pred_scores_sorted[:5000][val_pred_isin[:5000]], val_pred_scores_sorted[:5000][~val_pred_isin[:5000]])
        val_logit_distribution_plots.savefig(f"output/{args.dataset}_{args.task}/figures/5000_logit_distribution_{epoch}.png")
        plt.close(val_logit_distribution_plots)
        print("Generated val logit distribution plot")

model.load_state_dict(state_dict)

# val metrics
val_pred, val_pred_scores_sorted = test(*eval_loaders_dict["val"])
val_pred_isin, val_dst_count = get_pred_isin_and_dst_count(val_pred, task.get_table("val"), task.dst_entity_col)
val_auc = auc(val_pred_isin, val_dst_count)
val_map = task.evaluate(val_pred[:, :task.eval_k], task.get_table("val"))
print(f"Best Val auc: {val_auc}")
print(f"Best Val map: {val_map}")

# val plots
# ranking - first 2000 sources
val_ranking_plots = plot_rankings(val_pred_isin[:2000], task.eval_k)
val_ranking_plots.fig_scatter.savefig(f"output/{args.dataset}_{args.task}/figures/2000_ranking_{epoch}.png")
val_ranking_plots.fig_zoomed.savefig(f"output/{args.dataset}_{args.task}/figures/2000_ranking_zoomed_{epoch}.png")
plt.close(val_ranking_plots.fig_scatter)
plt.close(val_ranking_plots.fig_zoomed)
# ranking - all
val_ranking_plots = plot_rankings(val_pred_isin, task.eval_k)
val_ranking_plots.fig_scatter.savefig(f"output/{args.dataset}_{args.task}/figures/ranking_{epoch}.png")
val_ranking_plots.fig_zoomed.savefig(f"output/{args.dataset}_{args.task}/figures/ranking_zoomed_{epoch}.png")
plt.close(val_ranking_plots.fig_scatter)
plt.close(val_ranking_plots.fig_zoomed)
# logits distribution - all
val_logit_distribution_plots = plot_logit_distribution(val_pred_scores_sorted[val_pred_isin], val_pred_scores_sorted[~val_pred_isin])
val_logit_distribution_plots.savefig(f"output/{args.dataset}_{args.task}/figures/logit_distribution_{epoch}.png")
plt.close(val_logit_distribution_plots)


# test metrics
test_pred, test_pred_scores_sorted = test(*eval_loaders_dict["test"])
test_pred_isin, test_dst_count = get_pred_isin_and_dst_count(test_pred, task.get_table("test"), task.dst_entity_col)
test_auc = auc(test_pred_isin, test_dst_count)
test_map = task.evaluate(test_pred[:, :task.eval_k], task.get_table("test"))
print(f"Best test auc: {test_auc}")
print(f"Best test map: {test_map}")

# test plots
test_ranking_plots = plot_rankings(test_pred_isin, task.eval_k)
test_ranking_plots.fig_scatter.savefig(f"output/{args.dataset}_{args.task}/figures/ranking_{epoch}.png")
test_ranking_plots.fig_zoomed.savefig(f"output/{args.dataset}_{args.task}/figures/ranking_zoomed_{epoch}.png")
plt.close(test_ranking_plots.fig_scatter)
plt.close(test_ranking_plots.fig_zoomed)
test_logit_distribution_plots = plot_logit_distribution(test_pred_scores_sorted[test_pred_isin], test_pred_scores_sorted[~test_pred_isin])
test_logit_distribution_plots.savefig(f"output/{args.dataset}_{args.task}/figures/logit_distribution_{epoch}.png")
plt.close(test_logit_distribution_plots)
