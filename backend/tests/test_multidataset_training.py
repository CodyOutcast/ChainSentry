from app.ml.training.adaptors import EthLabelsAdaptor, FortaLabelAdaptor, PTXPhishAdaptor, RavenAdaptor
from app.ml.training.multi_dataset import build_split_dataset


def test_adaptors_emit_masked_samples_for_expected_heads() -> None:
    forta_samples = FortaLabelAdaptor().build_samples(split="train", limit=4, seed=11)
    eth_samples = EthLabelsAdaptor().build_samples(split="train", limit=2, seed=11)
    ptx_samples = PTXPhishAdaptor().build_samples(split="train", seed=11)
    raven_samples = RavenAdaptor().build_samples(split="train", limit=4, seed=11)

    assert any(sample.binary_target_mask["destination"] for sample in forta_samples)
    assert any(sample.binary_target_mask["approval"] for sample in forta_samples)
    assert all(sample.multiclass_target_mask["severity"] for sample in eth_samples)
    assert any(sample.binary_target_mask["simulation"] for sample in ptx_samples)
    assert all(sample.binary_target_mask["failure_aux"] for sample in raven_samples)
    assert all(not sample.multiclass_target_mask["severity"] for sample in raven_samples)


def test_multi_dataset_training_set_summarizes_mixed_sources() -> None:
    dataset = build_split_dataset(
        split="train",
        adaptors={
            "forta": FortaLabelAdaptor(),
            "eth-labels": EthLabelsAdaptor(),
        },
        limits={"forta": 3, "eth-labels": 2},
        seed=17,
    )

    summary = dataset.summarize("train")

    assert summary.total_samples == len(dataset)
    assert summary.dataset_counts["forta"] > 0
    assert summary.dataset_counts["eth-labels"] > 0
    assert summary.target_coverage["binary_mask_totals"]["destination"] > 0
