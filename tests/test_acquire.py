import numpy as np
from conftest import make_seed

from shrewd import acquire

MURKY = [  # every one mixes two intents, each worded differently
    "refund me, the export crashed", "cancel my plan, the invoice is wrong",
    "double charge after the update broke sync", "downgrade and refund the last charge",
    "error on checkout and I want to cancel", "broken dashboard, refund pending",
    "cancel the account, receipt shows an extra charge", "crash on login, downgrade please",
    "invoice error, considering cancellation", "charged twice while the app was broken",
]


def test_select_prefers_uncertain_rows():
    seed = make_seed(30)
    probe = acquire.fit_probe(seed["text"].tolist(), seed["label"].tolist(), 42)
    clear = [f"please cancel my subscription now #{i}" for i in range(30)]
    picked = acquire.select(probe, clear + MURKY, 10, 42)
    assert len(picked) == len(set(picked)) == 10
    assert sum(i >= 30 for i in picked) >= 6  # most of the batch is the mixed-intent rows


def test_select_does_not_fill_a_batch_with_near_duplicates():
    seed = make_seed(30)
    probe = acquire.fit_probe(seed["text"].tolist(), seed["label"].tolist(), 42)
    clear = [f"how do I change my avatar {i}" for i in range(30)]
    dupes = [f"charge crash cancel thing {i}" for i in range(10)]  # one hard case, ten times
    picked = acquire.select(probe, clear + dupes, 10, 42)
    assert 1 <= sum(i >= 30 for i in picked) <= 3


def test_select_returns_everything_when_batch_covers_the_pool():
    seed = make_seed(30)
    probe = acquire.fit_probe(seed["text"].tolist(), seed["label"].tolist(), 42)
    assert acquire.select(probe, ["a", "b", "c"], 5, 42) == [0, 1, 2]


def test_default_batch_scales_with_the_pool():
    assert acquire.default_batch(100) == 50
    assert acquire.default_batch(3000) == 300
    assert acquire.default_batch(100_000) == 500
    assert isinstance(acquire.default_batch(100), int) and not isinstance(np.int64(1), int)
