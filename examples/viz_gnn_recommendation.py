# %%
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from relbench.base import Dataset, RecommendationTask
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import LinkNeighborLoader
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig


# %%
def get_color(node_type: str, node_type_to_color: Dict[str, float]) -> float:
    """Get color for a node type."""
    if node_type not in node_type_to_color:
        node_type_to_color[node_type] = len(node_type_to_color) / max(1, len(node_type_to_color))
    return node_type_to_color[node_type]


def extract_nodes_and_edges(
    batch: HeteroData,
    source_node_type: NodeType,
    dst_node_type: NodeType,
) -> Tuple[Set[Tuple[NodeType, int]], List[Tuple[Tuple[NodeType, int], Tuple[NodeType, int], str]], Set[Tuple[NodeType, int]], Set[Tuple[NodeType, int]]]:
    """Extract nodes, edges, and identify source/destination nodes from a HeteroData batch.
    
    Returns:
        - all_nodes: Set of (node_type, node_idx) tuples
        - edges: List of ((src_type, src_idx), (dst_type, dst_idx), edge_type) tuples
        - source_nodes: Set of (node_type, node_idx) tuples for source nodes
        - dst_nodes: Set of (node_type, node_idx) tuples for destination nodes
    """
    all_nodes = set()
    edges = []
    source_nodes = set()
    dst_nodes = set()
    
    # Extract all nodes and identify input nodes
    for node_type in batch.node_types:
        if hasattr(batch[node_type], 'n_id'):
            n_id = batch[node_type].n_id
            for local_idx, global_idx in enumerate(n_id.tolist()):
                all_nodes.add((node_type, int(global_idx)))
        
        # Identify source nodes (input nodes in source batch)
        if node_type == source_node_type and hasattr(batch[node_type], 'input_id'):
            if hasattr(batch[node_type], 'n_id'):
                n_id = batch[node_type].n_id
                # input_id contains the global node indices that were used as input
                # We need to find which local positions correspond to these
                input_id = batch[node_type].input_id
                for input_global_idx in input_id.tolist():
                    # Find local position where n_id matches input_global_idx
                    local_positions = (n_id == input_global_idx).nonzero(as_tuple=True)[0]
                    for pos in local_positions:
                        source_nodes.add((node_type, int(n_id[pos])))
        
        # Identify destination nodes (input nodes in destination batch)
        if node_type == dst_node_type and hasattr(batch[node_type], 'input_id'):
            if hasattr(batch[node_type], 'n_id'):
                n_id = batch[node_type].n_id
                input_id = batch[node_type].input_id
                for input_global_idx in input_id.tolist():
                    local_positions = (n_id == input_global_idx).nonzero(as_tuple=True)[0]
                    for pos in local_positions:
                        dst_nodes.add((node_type, int(n_id[pos])))
    
    # Extract all edges
    for edge_type in batch.edge_types:
        src_type, edge_name, dst_type = edge_type
        if hasattr(batch[edge_type], 'edge_index'):
            edge_index = batch[edge_type].edge_index
            if edge_index.size(1) > 0:
                # Map local indices to global n_id
                src_n_id = batch[src_type].n_id
                dst_n_id = batch[dst_type].n_id
                
                for i in range(edge_index.size(1)):
                    src_local_idx = int(edge_index[0, i])
                    dst_local_idx = int(edge_index[1, i])
                    src_global_idx = int(src_n_id[src_local_idx])
                    dst_global_idx = int(dst_n_id[dst_local_idx])
                    
                    edges.append((
                        (src_type, src_global_idx),
                        (dst_type, dst_global_idx),
                        edge_name
                    ))
    
    return all_nodes, edges, source_nodes, dst_nodes


def viz(
    batch: HeteroData,
    dataset: Dataset,
    task: RecommendationTask,
    ax,
    dataset_name: str,
    title: str,
    source_nodes_set: Set[Tuple[NodeType, int]] = None,
    dst_nodes_set: Set[Tuple[NodeType, int]] = None,
    plot_labels: bool = False,
):
    """Visualize the context gathered by the GNN recommendation model for a single batch."""
    
    # Extract nodes and edges from the batch
    print(f"Extracting nodes and edges from batch...")
    all_nodes, edges, batch_source_nodes, batch_dst_nodes = extract_nodes_and_edges(
        batch, task.src_entity_table, task.dst_entity_table
    )
    print(f"Found {len(all_nodes)} nodes, {len(edges)} edges")
    
    # Use provided sets if available, otherwise use extracted ones
    if source_nodes_set is not None:
        source_nodes = source_nodes_set
    else:
        source_nodes = batch_source_nodes
    
    if dst_nodes_set is not None:
        dst_nodes = dst_nodes_set
    else:
        dst_nodes = batch_dst_nodes
    
    # Get table information for node type mapping
    # We'll use node types directly as table names
    node_type_to_color = {}
    
    # Build the graph
    g = nx.DiGraph()
    
    # Add all nodes
    for node_type, node_idx in all_nodes:
        # Create a unique node identifier
        node_id = f"{node_type}_{node_idx}"
        g.add_node(node_id, node_type=node_type, node_idx=node_idx)
    
    # Add all edges
    for (src_type, src_idx), (dst_type, dst_idx), edge_name in all_edges:
        src_id = f"{src_type}_{src_idx}"
        dst_id = f"{dst_type}_{dst_idx}"
        if src_id in g and dst_id in g:
            g.add_edge(src_id, dst_id, type=edge_name)
    
    # Categorize nodes
    source_node_ids = {f"{nt}_{idx}" for nt, idx in source_nodes if (nt, idx) in all_nodes}
    dst_node_ids = {f"{nt}_{idx}" for nt, idx in dst_nodes if (nt, idx) in all_nodes}
    
    # Get node positions
    if len(g.nodes()) > 0:
        print(f"Computing layout for {len(g.nodes())} nodes...")
        try:
            pos = nx.nx_agraph.graphviz_layout(nx.Graph(g), prog="twopi", root=list(g.nodes())[0])
        except Exception as e:
            # Fallback to spring layout if graphviz is not available
            print(f"Warning: graphviz layout failed ({e}), using spring layout")
            # Use a faster layout for large graphs
            if len(g.nodes()) > 1000:
                print("Using kamada_kawai layout for large graph")
                pos = nx.kamada_kawai_layout(g)
            else:
                pos = nx.spring_layout(g, k=1, iterations=50)
        print("Layout computed")
    else:
        pos = {}
        print("Warning: Empty graph, no nodes to visualize")
    
    cmap = plt.get_cmap("tab20")
    
    # Draw nodes by category
    db_nodes = []
    source_node_list = []
    dst_node_list = []
    
    for node_id in g.nodes():
        node_type = g.nodes[node_id]['node_type']
        color = get_color(node_type, node_type_to_color)
        
        if node_id in source_node_ids:
            source_node_list.append((node_id, node_type, color))
        elif node_id in dst_node_ids:
            dst_node_list.append((node_id, node_type, color))
        else:
            db_nodes.append((node_id, node_type, color))
    
    # Draw database nodes
    if db_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[node for node, _, _ in db_nodes],
            ax=ax,
            node_color=[color for _, _, color in db_nodes],
            cmap=cmap,
            vmin=0,
            vmax=1,
            node_size=100,
            node_shape="o",
            alpha=0.8,
        )
    
    # Draw source nodes
    if source_node_list:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[node for node, _, _ in source_node_list],
            ax=ax,
            node_color=[color for _, _, color in source_node_list],
            cmap=cmap,
            vmin=0,
            vmax=1,
            node_size=120,
            node_shape="s",
            alpha=0.8,
        )
    
    # Draw destination nodes
    if dst_node_list:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[node for node, _, _ in dst_node_list],
            ax=ax,
            node_color="green",
            node_size=150,
            node_shape="^",
        )
    
    # Draw edges
    f2p_edges = [(u, v) for u, v, data in g.edges(data=True) if "f2p" in data.get("type", "")]
    other_edges = [(u, v) for u, v, data in g.edges(data=True) if "f2p" not in data.get("type", "")]
    
    # Draw f2p edges
    if f2p_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=f2p_edges,
            ax=ax,
            edge_color=(0.2, 0.2, 0.2, 0.6),
            arrows=True,
            arrowsize=10,
            arrowstyle="->",
        )
    
    # Draw other edges
    if other_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=other_edges,
            ax=ax,
            edge_color=(0.5, 0.5, 0.5, 0.4),
            arrows=True,
            arrowsize=8,
            arrowstyle="->",
        )
    
    # Draw labels if requested
    if plot_labels:
        labels = {node: str(g.nodes[node]['node_idx']) for node in g.nodes()}
        nx.draw_networkx_labels(
            g,
            pos,
            labels,
            ax=ax,
            font_size=6,
            font_color="#301934",
        )
    
    # Create legend
    legend_handles: List[Union[Patch, Line2D]] = []
    table_color_map = {}
    
    for _, table_name, color in source_node_list:
        table_color_map[(table_name, True)] = color
    for _, table_name, color in db_nodes:
        table_color_map[(table_name, False)] = color
    
    for (table_name, is_source), color in sorted(table_color_map.items()):
        shape = "□" if is_source else "○"
        legend_handles.append(
            Patch(facecolor=cmap(color), label=f"{shape} {table_name}")
        )
    
    legend_handles.extend(
        [
            Patch(facecolor="none", label=""),
            Patch(label="○ Dataset nodes"),
            Patch(label="□ Source nodes"),
            Patch(label="▲ Destination nodes"),
            Line2D([0], [0], color=(0.2, 0.2, 0.2, 0.6), linewidth=2,
                   marker='>', markersize=8, label="→ F2P edges"),
            Line2D([0], [0], color=(0.5, 0.5, 0.5, 0.4), linewidth=2,
                   marker='>', markersize=8, label="→ Other edges"),
        ]
    )
    
    ax.legend(
        handles=legend_handles,
        title="Node Types & Tables",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    
    num_nodes = len(g.nodes())
    num_edges = len(g.edges())
    num_sources = len(source_node_ids)
    num_dst = len(dst_node_ids)
    ax.set_title(
        f"{title}\n"
        f"Graph: {num_nodes} nodes, {num_edges} edges\n"
        f"{num_sources} sources, {num_dst} destinations"
    )


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-trial")
    parser.add_argument("--task", type=str, default="condition-sponsor-run")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--plot_labels", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and task
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: RecommendationTask = get_task(args.dataset, args.task, download=True)
    
    # Get stypes
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
    
    # Build graph
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )
    
    # Create loader
    num_neighbors = [64, 32]  # Default values
    train_table_input = get_link_train_table_input(task.get_table("train"), task)
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        time_attr="time",
        src_nodes=train_table_input.src_nodes,
        dst_nodes=train_table_input.dst_nodes,
        num_dst_nodes=train_table_input.num_dst_nodes,
        src_time=train_table_input.src_time,
        share_same_time=True,
        batch_size=1,
        temporal_strategy="uniform",
        num_workers=0,
    )
    
    # Create output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Visualize batches
    for i in range(min(args.num_samples, len(train_loader))):
        batch_idx = args.batch_idx + i
        if batch_idx >= len(train_loader):
            break
        
        src_batch, pos_dst_batch, neg_dst_batch = train_loader[batch_idx]
        
        # Extract source and destination nodes from all batches
        _, _, source_nodes, _ = extract_nodes_and_edges(
            src_batch, task.src_entity_table, task.dst_entity_table
        )
        _, _, _, pos_dst_nodes = extract_nodes_and_edges(
            pos_dst_batch, task.src_entity_table, task.dst_entity_table
        )
        _, _, _, neg_dst_nodes = extract_nodes_and_edges(
            neg_dst_batch, task.src_entity_table, task.dst_entity_table
        )
        
        # Create figure with three subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), layout="constrained")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[2].set_axis_off()
        
        # Visualize source context
        viz(
            src_batch,
            dataset, task, axes[0],
            args.dataset,
            f"{args.dataset} {args.task} (Source context)",
            source_nodes_set=source_nodes,
            dst_nodes_set=None,
            plot_labels=args.plot_labels,
        )
        
        # Visualize positive destination context
        viz(
            pos_dst_batch,
            dataset, task, axes[1],
            args.dataset,
            f"{args.dataset} {args.task} (Positive destination context)",
            source_nodes_set=None,
            dst_nodes_set=pos_dst_nodes,
            plot_labels=args.plot_labels,
        )
        
        # Visualize negative destination context
        viz(
            neg_dst_batch,
            dataset, task, axes[2],
            args.dataset,
            f"{args.dataset} {args.task} (Negative destination context)",
            source_nodes_set=None,
            dst_nodes_set=neg_dst_nodes,
            plot_labels=args.plot_labels,
        )
        
        output_path = output_dir / f"{args.dataset}_{args.task}_batch_{batch_idx}.pdf"
        fig.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
