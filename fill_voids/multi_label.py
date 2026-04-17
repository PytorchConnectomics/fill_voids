"""
Multi-label void filling.

A void (a maximal connected region of ``0`` voxels) is filled with the
*outermost enclosing* foreground label -- defined as the adjacent
foreground label with the shortest chain of label-to-label adjacencies
back to the image exterior. Voids bounded by two or more labels that tie
for outermost (i.e. lie "in between" distinct shells) are left unfilled,
matching the semantics described in the project README.

This is the multi-label generalization of
``scipy.ndimage.binary_fill_holes``: when the input is effectively
binary it produces the same result as :func:`fill_voids.fill`; for a
densely labeled volume it produces the result one would get from the
naive loop ``for L in unique(labels): binary_fill(labels == L)``,
*but in a single sweep whose cost does not scale with the number of
labels* (the naive loop is O(K * N); this routine is O(N) for the
voxel-scale passes plus O(|label_graph|) for the small adjacency
analysis).

Algorithm
---------
1. Compute the 6- or 4-connected components of the background mask
   (``cc3d.connected_components``).
2. For each background component, collect the set of foreground labels
   that share a face with it by a single vectorized axis-shift scan.
3. Classify each component as exterior (touches the image boundary) or
   interior.
4. Compute ``level[L]``: the shortest number of label-to-label hops from
   ``L`` back to a label that directly touches the exterior. Shell
   labels (touching the image boundary or an exterior background
   component) have level 0. Unreachable labels (fully enclosed islands)
   have level infinity.
5. For each interior background component:
     * If it is adjacent to exactly one foreground label, fill with it.
     * Otherwise, pick the adjacent label with the smallest ``level``.
       If that minimum is shared by two or more labels, leave the
       component unfilled (it lies between distinct shells).
6. Apply the fill via a single ``lut[bg_cc]`` lookup.

Author: extension for the ``fill_voids`` library, 2026-04-16.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional, Tuple, Union

import numpy as np


__all__ = ["fill_multi_label"]


_INF = np.iinfo(np.int64).max


def fill_multi_label(
    labels: np.ndarray,
    in_place: bool = False,
    return_fill_count: bool = False,
    connectivity: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Fill voids in a 2D or 3D multi-labeled image.

    Parameters
    ----------
    labels:
        Integer array. Voxels equal to ``0`` are background (candidate
        voids); any other value is a foreground label.
    in_place:
        If ``True``, modify ``labels`` in place. Otherwise operate on a
        copy.
    return_fill_count:
        If ``True``, also return the number of background voxels filled.
    connectivity:
        Neighborhood connectivity. ``6`` (default in 3D) or ``26`` for
        3D; ``4`` (default in 2D) or ``8`` for 2D. Face connectivity
        (6/4) matches :func:`fill_voids.fill` and
        ``scipy.ndimage.binary_fill_holes``.
    """
    try:
        import cc3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fill_voids.fill_multi_label requires connected-components-3d "
            "(`pip install connected-components-3d`)."
        ) from exc

    if labels.ndim not in (2, 3):
        raise ValueError(
            f"fill_multi_label requires 2D or 3D input, got {labels.ndim}D"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(
            f"fill_multi_label requires an integer dtype, got {labels.dtype}"
        )

    if not in_place:
        labels = labels.copy()

    if labels.size == 0:
        return (labels, 0) if return_fill_count else labels

    ndim = labels.ndim
    if connectivity is None:
        connectivity = 6 if ndim == 3 else 4
    _validate_connectivity(ndim, connectivity)

    bg_mask = labels == 0
    if not bg_mask.any():
        return (labels, 0) if return_fill_count else labels

    # 1. Connected components of the background.
    bg_cc, num_cc = cc3d.connected_components(
        bg_mask, connectivity=connectivity, return_N=True
    )
    if num_cc == 0:
        return (labels, 0) if return_fill_count else labels

    # 2. Which BG components touch the image boundary?
    cc_on_boundary = np.zeros(num_cc + 1, dtype=bool)
    lbl_on_boundary: "set[int]" = set()
    for axis in range(ndim):
        for idx in (0, labels.shape[axis] - 1):
            cc_face = np.take(bg_cc, idx, axis=axis)
            lbl_face = np.take(labels, idx, axis=axis)
            cc_on_boundary[np.unique(cc_face)] = True
            for v in np.unique(lbl_face):
                if v != 0:
                    lbl_on_boundary.add(int(v))
    cc_on_boundary[0] = False

    interior_cc = np.flatnonzero(~cc_on_boundary)
    interior_cc = interior_cc[interior_cc > 0]
    if interior_cc.size == 0:
        return (labels, 0) if return_fill_count else labels

    # 3. BG-component <-> foreground-label adjacency via a single
    #    vectorized axis-shift scan. For each pair of adjacent voxels
    #    along every axis in both directions, a (cc_id, label) pair is
    #    emitted when one side is a BG voxel and the other a label.
    pair_chunks = []
    for axis in range(ndim):
        s1 = _axis_slice(ndim, axis, slice(None, -1))
        s2 = _axis_slice(ndim, axis, slice(1, None))
        cc_a, cc_b = bg_cc[s1], bg_cc[s2]
        lbl_a, lbl_b = labels[s1], labels[s2]
        _append_cc_label_pairs(pair_chunks, cc_a, lbl_b)
        _append_cc_label_pairs(pair_chunks, cc_b, lbl_a)
        if connectivity in (8, 18, 26):
            _append_diagonal_pairs(pair_chunks, bg_cc, labels, axis, ndim)

    if not pair_chunks:
        return (labels, 0) if return_fill_count else labels

    cc_pairs = np.concatenate(pair_chunks, axis=0)
    cc_ids, lbl_ids = _unique_pairs(cc_pairs[:, 0], cc_pairs[:, 1])

    # Group by cc_id so we can quickly query "labels adjacent to CC c".
    order = np.argsort(cc_ids, kind="stable")
    cc_ids = cc_ids[order]
    lbl_ids = lbl_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(cc_ids)), cc_ids.size]
    cc_adj: "dict[int, np.ndarray]" = {}
    for i in range(starts.size - 1):
        cc_adj[int(cc_ids[starts[i]])] = lbl_ids[starts[i]:starts[i + 1]]

    # 4. Shell labels: those that can reach the exterior without
    #    crossing another label. Level 0 in the BFS below.
    shell_labels = set(lbl_on_boundary)
    for c in np.flatnonzero(cc_on_boundary):
        c = int(c)
        if c == 0:
            continue
        adj = cc_adj.get(c)
        if adj is not None:
            shell_labels.update(int(v) for v in adj)

    # 5. Label-to-label adjacency via cc3d.region_graph on the original
    #    labels array (ignoring background). This is cheap: the output
    #    is one edge per (distinct) pair of labels that share a face.
    lbl_adj: "defaultdict[int, list[int]]" = defaultdict(list)
    for a, b in cc3d.region_graph(labels, connectivity=connectivity):
        a, b = int(a), int(b)
        if a == 0 or b == 0:
            continue
        lbl_adj[a].append(b)
        lbl_adj[b].append(a)

    # 6. BFS to compute level[L] = shortest label-chain distance to the
    #    exterior. Shell labels seed the BFS at level 0.
    level: "dict[int, int]" = {L: 0 for L in shell_labels}
    q = deque(shell_labels)
    while q:
        cur = q.popleft()
        cur_level = level[cur] + 1
        for nb in lbl_adj.get(cur, ()):
            if nb not in level:
                level[nb] = cur_level
                q.append(nb)

    # 7. For each interior BG component, decide its fill label.
    lut = np.zeros(num_cc + 1, dtype=labels.dtype)
    for c in interior_cc:
        c = int(c)
        adj = cc_adj.get(c)
        if adj is None or adj.size == 0:
            continue
        if adj.size == 1:
            lut[c] = adj[0]
            continue
        # Smallest level wins; a tie means the void is between shells.
        levels = np.fromiter(
            (level.get(int(L), _INF) for L in adj),
            dtype=np.int64,
            count=adj.size,
        )
        min_lvl = levels.min()
        winners = adj[levels == min_lvl]
        if winners.size == 1:
            lut[c] = winners[0]
        # else: ambiguous -> leave as 0 (no fill).

    # 8. Apply the fill in a single pass.
    fill_arr = lut[bg_cc]
    fill_mask = fill_arr != 0
    num_filled = int(np.count_nonzero(fill_mask))
    if num_filled:
        labels[fill_mask] = fill_arr[fill_mask]

    return (labels, num_filled) if return_fill_count else labels


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_connectivity(ndim: int, connectivity: int) -> None:
    valid = {2: (4, 8), 3: (6, 18, 26)}[ndim]
    if connectivity not in valid:
        raise ValueError(
            f"connectivity={connectivity} invalid for {ndim}D; "
            f"use one of {valid}"
        )


def _axis_slice(ndim: int, axis: int, s: slice) -> tuple:
    out = [slice(None)] * ndim
    out[axis] = s
    return tuple(out)


def _append_cc_label_pairs(chunks: list, cc_side: np.ndarray, lbl_side: np.ndarray) -> None:
    m = (cc_side != 0) & (lbl_side != 0)
    if m.any():
        chunks.append(np.column_stack((cc_side[m], lbl_side[m])))


def _append_diagonal_pairs(chunks: list, bg_cc: np.ndarray, labels: np.ndarray,
                           axis: int, ndim: int) -> None:
    # For diagonal connectivity (8/18/26), also scan edges/corners. Keep
    # this a thin fallback: for the common face-connectivity path it is
    # not called. The scan emits pairs along every (axis, other_axis)
    # offset pair.
    other_axes = [a for a in range(ndim) if a != axis]
    for other in other_axes:
        for s1_dir, s2_dir in ((slice(None, -1), slice(1, None)),
                               (slice(1, None), slice(None, -1))):
            s1 = _axis_slice(ndim, axis, slice(None, -1))
            s1 = tuple(slice(None, -1) if i == axis else (s1_dir if i == other else slice(None))
                       for i in range(ndim))
            s2 = tuple(slice(1, None) if i == axis else (s2_dir if i == other else slice(None))
                       for i in range(ndim))
            cc_a, cc_b = bg_cc[s1], bg_cc[s2]
            lbl_a, lbl_b = labels[s1], labels[s2]
            _append_cc_label_pairs(chunks, cc_a, lbl_b)
            _append_cc_label_pairs(chunks, cc_b, lbl_a)


def _unique_pairs(col0: np.ndarray, col1: np.ndarray) -> tuple:
    """Return the unique (col0, col1) pairs.

    Uses bit-packing into uint64 when values fit in 32 bits, which is
    far faster than ``np.unique(axis=0)`` on a 2-column array.
    """
    c0 = col0.astype(np.int64, copy=False)
    c1 = col1.astype(np.int64, copy=False)
    if c0.max() < (1 << 32) and c1.max() < (1 << 32) and c0.min() >= 0 and c1.min() >= 0:
        enc = (c0.astype(np.uint64) << np.uint64(32)) | c1.astype(np.uint64)
        uniq = np.unique(enc)
        out0 = (uniq >> np.uint64(32)).astype(np.int64)
        out1 = (uniq & np.uint64(0xFFFFFFFF)).astype(np.int64)
        return out0, out1
    # Fallback: slower but always correct.
    pairs = np.column_stack((c0, c1))
    uniq = np.unique(pairs, axis=0)
    return uniq[:, 0], uniq[:, 1]
