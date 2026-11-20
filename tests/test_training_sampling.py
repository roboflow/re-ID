# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``PKSampler``: batches of P identities with K crops each, and the per-sequence groups it can draw from."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from reid.training.sampling import PKSampler, build_sequence_groups


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


def test_every_batch_comes_from_one_group() -> None:
    labels = [identity for identity in range(8) for _ in range(2)]
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]  # such as the identities of two sequences

    sampler = PKSampler(labels, identities_per_batch=2, crops_per_identity=2, num_batches=50, seed=0, groups=groups)

    for batch in sampler:
        identities = {labels[index] for index in batch}
        assert identities <= set(groups[0]) or identities <= set(groups[1])


def test_a_small_group_gives_a_smaller_batch_of_all_its_identities() -> None:
    labels = [0, 0, 1, 1]

    sampler = PKSampler(labels, identities_per_batch=4, crops_per_identity=2, num_batches=20, seed=0, groups=[[0, 1]])

    for batch in sampler:
        assert sorted(batch) == [0, 1, 2, 3]  # both identities of the group, 2 crops each, instead of 4 identities


def test_an_identity_of_a_small_group_is_drawn_as_often_as_one_of_a_large_group() -> None:
    labels = [identity for identity in range(10) for _ in range(2)]
    groups = [[0, 1], [2, 3, 4, 5, 6, 7, 8, 9]]

    sampler = PKSampler(labels, identities_per_batch=4, crops_per_identity=2, num_batches=3000, seed=0, groups=groups)

    visits = Counter(identity for batch in sampler for identity in {labels[index] for index in batch})

    # the group of 2 is drawn with weight max(2, 4) = 4 and gives both its identities: 4 / 12 x 1 = 1 / 3 per batch,
    # and the group of 8 with weight max(8, 4) = 8 and gives 4 of its 8: 8 / 12 x 4 / 8 = 1 / 3 per batch
    assert visits[0] == pytest.approx(1000, rel=0.1)  # 3000 batches / 3
    assert visits[2] == pytest.approx(1000, rel=0.1)


def test_groups_the_identities_of_each_sequence() -> None:
    image_paths = ["a_0/a_000000.jpg", "b_1/b_000000.jpg", "a_2/a_000000.jpg", "b_3/b_000000.jpg"]

    groups = build_sequence_groups(image_paths, labels=np.array([0, 1, 2, 3]))

    assert groups == [[0, 2], [1, 3]]  # identities 0 and 2 are in sequence a, 1 and 3 in b


def test_leaves_out_a_sequence_with_a_single_identity_with_a_warning() -> None:
    image_paths = ["a_0/a_000000.jpg", "a_1/a_000000.jpg", "b_2/b_000000.jpg"]

    with pytest.warns(UserWarning, match="Sequences with a single identity will be left out.*: b."):
        groups = build_sequence_groups(image_paths, labels=np.array([0, 1, 2]))

    assert groups == [[0, 1]]  # identity 2 is alone in sequence b
