import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from datasets.graph_datasets.graph_heat_dataset import HeatGraphDataset

# ---------------------------------------------------------------------------
# Fixtures
#
# The synthetic field encodes both the frame index and the grid position into
# every value:   X[n, t, a, b] = t * FRAME_STRIDE + (a * W + b)
# so any measurement read out of a sample can be inverted back to the node it
# came from. That is what lets the alignment tests below check that edge
# endpoints, edge features and labels all refer to the same node ordering
# without needing to know which nodes the dataset happened to sample.
# ---------------------------------------------------------------------------

N_SIMS = 3
N_FRAMES = 4
GRID = 8
FRAME_STRIDE = 1000
SUB_GRAPH_SIZE = 24
RADIUS = 0.35


def _node_code(grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
    """Positional code embedded in the synthetic field.

    :param grid_x: Row indices.
    :type grid_x: torch.Tensor
    :param grid_y: Column indices.
    :type grid_y: torch.Tensor
    :returns: Unique integer identifying each grid cell.
    :rtype: torch.Tensor
    """
    return grid_x * GRID + grid_y


@pytest.fixture(scope="module")
def aggregated_path(tmp_path_factory) -> str:
    """Write a synthetic aggregated dataset to disk.

    :returns: Path to the saved .pt file.
    :rtype: str
    """
    a = torch.arange(GRID).view(-1, 1).expand(GRID, GRID)
    b = torch.arange(GRID).view(1, -1).expand(GRID, GRID)
    pos_code = _node_code(a, b).to(torch.float32)

    frames = torch.arange(N_FRAMES, dtype=torch.float32).view(1, N_FRAMES, 1, 1) * FRAME_STRIDE
    X = frames + pos_code.view(1, 1, GRID, GRID)
    X = X.expand(N_SIMS, N_FRAMES, GRID, GRID).contiguous()

    path = tmp_path_factory.mktemp("graph_data") / "aggregated.pt"
    torch.save({"X": X, "alpha": torch.rand(N_SIMS)}, path)
    return str(path)


@pytest.fixture
def dataset(aggregated_path: str) -> HeatGraphDataset:
    """A dataset over the synthetic field.

    :returns: Configured dataset instance.
    :rtype: HeatGraphDataset
    """
    return HeatGraphDataset(
        aggregated_path=aggregated_path,
        field_keys=["alpha"],
        r=RADIUS,
        bc="periodic",
        sub_graph_size=SUB_GRAPH_SIZE,
    )


# ---------------------------------------------------------------------------
# Construction and indexing
# ---------------------------------------------------------------------------

def test_len_reserves_a_next_frame(dataset: HeatGraphDataset):
    """Every index must have a t+1 frame available to read labels from.

    :returns: None
    """
    assert len(dataset) == N_SIMS * (N_FRAMES - 1)


def test_last_index_is_reachable(dataset: HeatGraphDataset):
    """The final index must not run off the end of the frame axis.

    :returns: None
    """
    dataset[len(dataset) - 1]


def test_sub_graph_size_validated(aggregated_path: str):
    """A sub_graph_size larger than the node count is rejected at construction.

    :returns: None
    """
    with pytest.raises(ValueError):
        HeatGraphDataset(aggregated_path, ["alpha"], RADIUS, "periodic", GRID * GRID + 1)


# ---------------------------------------------------------------------------
# edge_index structure
# ---------------------------------------------------------------------------

def test_edge_index_is_two_by_e(dataset: HeatGraphDataset):
    """PyG requires edge_index laid out as [2, E], not [E, 2].

    :returns: None
    """
    data = dataset[0]
    assert data.edge_index.shape[0] == 2, f"expected [2, E], got {tuple(data.edge_index.shape)}"


def test_edge_index_within_node_range(dataset: HeatGraphDataset):
    """Edge endpoints must be local indices into the m sampled nodes.

    :returns: None
    """
    data = dataset[0]
    assert int(data.edge_index.min()) >= 0
    assert int(data.edge_index.max()) < SUB_GRAPH_SIZE


def test_no_self_loops(dataset: HeatGraphDataset):
    """The radius graph drops self-loops.

    :returns: None
    """
    data = dataset[0]
    assert not bool((data.edge_index[0] == data.edge_index[1]).any())


def test_edge_attr_rows_match_edge_count(dataset: HeatGraphDataset):
    """One edge feature row per edge.

    :returns: None
    """
    data = dataset[0]
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]


def test_edge_attr_width(dataset: HeatGraphDataset):
    """Edge features are [disp_x, disp_y, u_src, u_dst, *pde_params].

    :returns: None
    """
    data = dataset[0]
    assert data.edge_attr.shape[1] == 2 + 2 + len(dataset.field_keys)


def test_displacements_within_radius(dataset: HeatGraphDataset):
    """Every edge displacement must lie inside the connection radius.

    :returns: None
    """
    data = dataset[0]
    assert bool((data.edge_attr[:, :2].norm(dim=-1) <= RADIUS + 1e-6).all())


# ---------------------------------------------------------------------------
# Node / label alignment
# ---------------------------------------------------------------------------

def test_label_count_matches_node_count(dataset: HeatGraphDataset):
    """n nodes must yield exactly n labels.

    :returns: None
    """
    data = dataset[0]
    assert data.y.shape[0] == SUB_GRAPH_SIZE


def test_label_is_column_vector(dataset: HeatGraphDataset):
    """Labels must be (m, 1) so an (m, 1) prediction compares elementwise.

    A flat (m,) target broadcasts against an (m, 1) prediction to (m, m),
    which silently produces a wrong loss rather than an error.

    :returns: None
    """
    data = dataset[0]
    assert data.y.shape == (SUB_GRAPH_SIZE, 1), f"got {tuple(data.y.shape)}"


def test_num_nodes_matches_sub_graph_size(dataset: HeatGraphDataset):
    """num_nodes must equal m so batching offsets edge_index correctly.

    :returns: None
    """
    data = dataset[0]
    assert data.num_nodes == SUB_GRAPH_SIZE


def test_labels_are_read_from_the_next_frame(dataset: HeatGraphDataset):
    """Labels come from t+1, not from t.

    :returns: None
    """
    index = 1
    frame_idx = index % (N_FRAMES - 1)
    codes = dataset[index].y.flatten() - (frame_idx + 1) * FRAME_STRIDE
    assert bool(((codes >= 0) & (codes < GRID * GRID)).all()), "labels not drawn from frame t+1"


def test_labels_reference_distinct_nodes(dataset: HeatGraphDataset):
    """Sampling is without replacement, so every label is a distinct node.

    :returns: None
    """
    frame_idx = 0
    codes = dataset[0].y.flatten() - (frame_idx + 1) * FRAME_STRIDE
    assert len(set(codes.tolist())) == SUB_GRAPH_SIZE


def test_edge_endpoints_index_the_same_nodes_as_labels(dataset: HeatGraphDataset):
    """The core alignment invariant.

    Recovers each node's identity from its label, then checks that the source
    and destination measurements carried in edge_attr are exactly the frame-t
    values of the nodes named by edge_index. This fails if edge_index, the
    edge features and y are built over different node orderings.

    :returns: None
    """
    index = 0
    frame_idx = index % (N_FRAMES - 1)
    data = dataset[index]

    codes = data.y.flatten() - (frame_idx + 1) * FRAME_STRIDE
    src, dst = data.edge_index[0], data.edge_index[1]

    expected_src = frame_idx * FRAME_STRIDE + codes[src]
    expected_dst = frame_idx * FRAME_STRIDE + codes[dst]

    assert torch.equal(data.edge_attr[:, 2], expected_src), "source measurement misaligned"
    assert torch.equal(data.edge_attr[:, 3], expected_dst), "destination measurement misaligned"


def test_labels_are_not_the_input(dataset: HeatGraphDataset):
    """The target must differ from the u(x, t) already present in edge_attr.

    :returns: None
    """
    data = dataset[0]
    assert not bool((data.edge_attr[:, 2] == data.y.flatten()[data.edge_index[0]]).all())


# ---------------------------------------------------------------------------
# Sampling behaviour
# ---------------------------------------------------------------------------

def test_sampling_varies_between_draws(dataset: HeatGraphDataset):
    """Repeated reads of the same index draw different node subsets.

    :returns: None
    """
    torch.manual_seed(0)
    first = set((dataset[0].y.flatten() - FRAME_STRIDE).tolist())
    second = set((dataset[0].y.flatten() - FRAME_STRIDE).tolist())
    assert first != second


def test_sampling_covers_the_whole_grid(dataset: HeatGraphDataset):
    """Nodes are drawn from the full domain, not a fixed corner.

    :returns: None
    """
    torch.manual_seed(0)
    seen = set()
    for _ in range(40):
        seen.update((dataset[0].y.flatten() - FRAME_STRIDE).tolist())
    assert max(seen) > SUB_GRAPH_SIZE, "sampled nodes confined to the first m indices"


def test_no_isolated_nodes(dataset: HeatGraphDataset):
    """Every sampled node has at least one neighbour at this r and m.

    A node with degree zero receives no messages and contributes an
    unlearnable target row.

    :returns: None
    """
    data = dataset[0]
    degree = torch.bincount(data.edge_index[0], minlength=SUB_GRAPH_SIZE)
    assert int(degree.min()) > 0, f"{int((degree == 0).sum())} isolated nodes at r={RADIUS}"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_batch_label_count_matches_nodes(dataset: HeatGraphDataset):
    """Batched labels stay one-per-node across the disjoint union.

    :returns: None
    """
    batch = Batch.from_data_list([dataset[0], dataset[1], dataset[2]])
    assert batch.y.shape[0] == batch.num_nodes == 3 * SUB_GRAPH_SIZE


def test_batch_has_a_node_assignment_vector(dataset: HeatGraphDataset):
    """PyG must be able to build the batch vector from the Data fields.

    :returns: None
    """
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    assert batch.batch is not None
    assert batch.batch.shape[0] == 2 * SUB_GRAPH_SIZE


def test_batching_does_not_create_cross_graph_edges(dataset: HeatGraphDataset):
    """edge_index offsetting must keep every edge inside its own subgraph.

    :returns: None
    """
    batch = Batch.from_data_list([dataset[0], dataset[1], dataset[2]])
    assert torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]])


def test_dataloader_round_trip(dataset: HeatGraphDataset):
    """The dataset works through the PyG DataLoader end to end.

    :returns: None
    """
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch.num_graphs == 2
    assert batch.y.shape[0] == 2 * SUB_GRAPH_SIZE
    assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
