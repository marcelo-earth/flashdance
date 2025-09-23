"""Pytest configuration for flashdance tests."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests when CUDA is not available."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def dtype(device):
    if device == "cpu":
        return torch.float32
    return torch.float32  # use float32 for reproducibility in tests


@pytest.fixture
def small_qkv(device):
    """Small Q, K, V tensors for fast tests."""
    B, H, T, D = 2, 4, 32, 64
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    return q, k, v


@pytest.fixture
def medium_qkv(device):
    """Medium Q, K, V tensors."""
    B, H, T, D = 4, 8, 256, 64
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    return q, k, v
