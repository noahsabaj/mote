"""Target-offset augmentation (2606.16246): the offset is read back off each window's own marker. A MixedShard
draws every row from its own shard with its own offset, so rows of one batch can differ — found 2026-08-29
(the trainer read row 0's digit for the whole batch)."""

import torch

from mote.tokenizer import OFFSET_ID
from mote.train.train import _targets


def test_rows_with_different_offsets_get_their_own_targets():
    L = 12
    base = torch.arange(100, 100 + L + 1)[None].repeat(3, 1)
    b = base.clone()
    b[0, 0], b[0, 1] = OFFSET_ID, ord("3")  # row 0 predicts x_{t+3}
    b[1, 0], b[1, 1] = OFFSET_ID, ord("2")  # row 1 predicts x_{t+2}
    # row 2: plain next-byte
    inputs, targets, tmask = _targets(b, None)
    assert torch.equal(inputs, b[:, :-1])
    assert torch.equal(targets[0, :L - 2], b[0, 3 : L + 1])  # x_{t+3}
    assert torch.equal(targets[1, :L - 1], b[1, 2 : L + 1])  # x_{t+2}
    assert torch.equal(targets[2], b[2, 1:])  # x_{t+1}
    assert tmask[0, L - 2 :].sum() == 0 and tmask[0, : L - 2].sum() == L - 2
    assert tmask[1, L - 1 :].sum() == 0 and tmask[2].sum() == L


def test_plain_batches_are_untouched():
    b = torch.randint(0, 256, (2, 9))
    inputs, targets, tmask = _targets(b, None)
    assert torch.equal(inputs, b[:, :-1]) and torch.equal(targets, b[:, 1:]) and tmask is None
