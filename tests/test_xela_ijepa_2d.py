import torch

from tactile_ssl.data.xela.atlas import (
    XELA_ATLAS_HEIGHT,
    XELA_ATLAS_WIDTH,
    xela_2d_sincos_position_embedding,
    xela_atlas_coordinates,
)
from tactile_ssl.model.xela_transformer import XelaTransformer
from tactile_ssl.utils.xela_ijepa_masking import XelaIJEPA2DMaskCollator


def test_xela_atlas_is_unique_and_complete() -> None:
    coordinates = xela_atlas_coordinates()
    assert coordinates.shape == (368, 2)
    assert torch.unique(coordinates, dim=0).shape[0] == 368
    assert coordinates[:, 0].max() < XELA_ATLAS_HEIGHT
    assert coordinates[:, 1].max() < XELA_ATLAS_WIDTH


def test_fixed_2d_embedding_is_deterministic() -> None:
    first = xela_2d_sincos_position_embedding(192)
    second = xela_2d_sincos_position_embedding(192)
    assert first.shape == (1, 368, 192)
    assert torch.equal(first, second)


def test_xela_transformer_uses_fixed_2d_embedding() -> None:
    model = XelaTransformer(
        in_dim=368,
        in_chans=3,
        time_chunk_size=10,
        sequence_length=10,
        embed_dim=192,
        depth=1,
        num_heads=3,
        pos_embed_fn="xela_2d_sincos",
        use_taxel_type_embedding=False,
    )
    assert not isinstance(model.pos_embed, torch.nn.Parameter)
    assert torch.equal(model.get_position_embedding(torch.device("cpu")), model.pos_embed)


def test_ijepa_collator_returns_one_context_and_four_rectangular_targets() -> None:
    torch.manual_seed(42)
    collator = XelaIJEPA2DMaskCollator(
        context_mask_scale=(0.5, 0.9),
        target_mask_scale=(0.1, 0.18),
        num_context_masks=1,
        num_target_masks=4,
    )
    samples = [{"sensor": torch.randn(10, 368, 3)} for _ in range(3)]
    batch = collator(samples)
    assert batch["context_masks"].shape[:2] == (1, 3)
    assert batch["target_masks"].shape[:2] == (4, 3)
    for batch_id in range(3):
        context = set(batch["context_masks"][0, batch_id].tolist())
        targets = set(batch["target_masks"][:, batch_id].flatten().tolist())
        assert context.isdisjoint(targets)
