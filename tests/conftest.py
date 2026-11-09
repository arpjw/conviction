import pytest

from conviction.data.splits import DataConfig, generate_dataset


@pytest.fixture(scope="session")
def small_cfg():
    return DataConfig(seed=7, n_indist_groups=240, n_template_groups=40, n_structural_groups=40)


@pytest.fixture(scope="session")
def small_data(small_cfg):
    return generate_dataset(small_cfg)


@pytest.fixture(scope="session")
def all_eps(small_data):
    return [e for eps in small_data.values() for e in eps]
