import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.nn import WLConvContinuous

from tactile_ssl.downstream_task.force_sl import XelaForceSpatialWLMLPProbe
from tactile_ssl.downstream_task.spatial_wl_mlp import XelaSpatialWLMLPEncoder
from tactile_ssl.downstream_task.xela_object import XelaObjectSpatialWLMLPClassifier
from tactile_ssl.downstream_task.xela_relativepose import XelaRelativePoseSpatialWLMLPDecoder
from tactile_ssl.model.xela_spatial_gnn import physical_graph_edge_pairs_torch


def _positions(num_graphs=2):
    generator = torch.Generator().manual_seed(7)
    return torch.randn(num_graphs, 368, 3, generator=generator)


def _static_graph_info(positions):
    edge_count = 2 * len(physical_graph_edge_pairs_torch(positions[0], bridge_k=4))
    return {"edge_count": torch.full((positions.shape[0],), edge_count, dtype=torch.long)}


def test_two_wl_layers_match_manual_chain_diffusion():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    features = torch.tensor([[0.0], [2.0], [4.0]])
    layer = WLConvContinuous()

    once = layer(features, edge_index)
    twice = layer(once, edge_index)

    torch.testing.assert_close(once, torch.tensor([[1.0], [2.0], [3.0]]))
    torch.testing.assert_close(twice, torch.tensor([[1.5], [2.0], [2.5]]))


def test_wl_mlp_encoder_static_graph_output_contract():
    positions = _positions()
    graph_info = _static_graph_info(positions)
    encoder = XelaSpatialWLMLPEncoder(spatial_hidden_dims=[32, 64], wl_num_layers=2)

    output = encoder(positions, graph_info)

    assert output.shape == (2, 368, 64)
    assert encoder.static_physical_edge_index.shape[0] == 2
    assert encoder.static_physical_edge_index.shape[1] == graph_info["edge_count"][0]


def test_force_and_pose_wl_fusion_preserve_192_dimensional_tokens():
    positions = _positions()
    graph_info = _static_graph_info(positions)
    signal_tokens = torch.randn(1, 2, 368, 192)
    spatial_coords = positions.view(1, 2, 368, 3)

    force_probe = XelaForceSpatialWLMLPProbe(
        time_chunk_size=10,
        embed_dim="tiny",
        num_heads=3,
        spatial_hidden_dims=[32, 64],
        spatial_wl_layers=2,
    )
    pose_decoder = XelaRelativePoseSpatialWLMLPDecoder(
        embed_dim="tiny",
        num_heads=3,
        spatial_hidden_dims=[32, 64],
        spatial_wl_layers=2,
    )

    force_tokens = force_probe._prepare_tokens(signal_tokens, spatial_coords, graph_info)
    pose_tokens = pose_decoder._prepare_tokens(signal_tokens, spatial_coords, graph_info)

    assert force_tokens.shape == signal_tokens.shape
    assert pose_tokens.shape == signal_tokens.shape


def test_object_wl_classifier_output_shape():
    positions = _positions(num_graphs=2)
    graph_info = _static_graph_info(positions)
    classifier = XelaObjectSpatialWLMLPClassifier(
        input_embed_dim=192,
        classes=["a", "b", "c"],
        spatial_hidden_dims=[32, 64],
        spatial_wl_layers=2,
    )

    output = classifier(
        torch.randn(2, 192),
        spatial_coords=positions.unsqueeze(1).repeat(1, 10, 1, 1),
        graph_info=graph_info,
    )

    assert output.shape == (2, 3)


def test_wl_mlp_encoder_validates_graph_and_coordinate_contracts():
    positions = _positions()
    encoder = XelaSpatialWLMLPEncoder(spatial_hidden_dims=[32, 64], wl_num_layers=2)

    with pytest.raises(ValueError, match="graph_info is required"):
        encoder(positions, None)
    with pytest.raises(ValueError, match="368 Xela nodes"):
        encoder(positions[:, :-1], {"edge_count": torch.ones(2, dtype=torch.long)})
    with pytest.raises(ValueError, match="same number of graphs"):
        encoder(positions, {"edge_count": torch.ones(1, dtype=torch.long)})
