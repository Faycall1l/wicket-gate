import pytest

from network_orch.cascade import llm


def test_configure_rejects_unknown_policy():
    with pytest.raises(ValueError):
        llm.configure("nonsense")
    llm.configure("auto")


def test_pinned_backend_does_not_fall_back(monkeypatch):
    llm.configure("primary")
    monkeypatch.setattr(llm.SETTINGS.__class__, "llm_endpoint",
                        lambda self: ("http://127.0.0.1:9", "k", "m"))
    with pytest.raises(RuntimeError, match="policy=primary"):
        llm.chat("s", "u", timeout=1.0)
    llm.configure("auto")
