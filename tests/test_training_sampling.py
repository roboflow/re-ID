# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``PKSampler``: batches of P identities with K crops each."""

from __future__ import annotations

from collections import Counter

import pytest

from reid.training.sampling import PKSampler


def test_crops_repeat_only_for_an_identity_with_fewer_than_k() -> None:
    labels = [0, 0, 0, 1, 1, 1, 2]  # identity 2 has one crop, fewer than K

    sampler = PKSampler(labels, identities_per_batch=3, crops_per_identity=3, num_batches=20, seed=0)

    for batch in sampler:
        assert sorted(batch) == [0, 1, 2, 3, 4, 5, 6, 6, 6]  # every crop of 0 and 1 once, identity 2's crop 3 times


def test_an_identity_with_more_crops_is_not_drawn_more_often() -> None:
    labels = [0] * 30 + [1, 1, 2, 2, 3, 3]  # identity 0 has 15 times the crops of the others

    sampler = PKSampler(labels, identities_per_batch=2, crops_per_identity=2, num_batches=2000, seed=0)

    visits = Counter(identity for batch in sampler for identity in {labels[index] for index in batch})

    assert visits[0] == pytest.approx(1000, rel=0.1)  # 2000 batches x 2 identities, over 4 identities


def test_each_epoch_draws_new_batches() -> None:
    labels = [identity for identity in range(8) for _ in range(4)]
    sampler = PKSampler(labels, identities_per_batch=2, crops_per_identity=2, num_batches=4, seed=0)

    # train iterates the same sampler once per epoch, and its random generator continues across epochs
    first_epoch = [batch for batch in sampler]
    second_epoch = [batch for batch in sampler]

    assert first_epoch != second_epoch
