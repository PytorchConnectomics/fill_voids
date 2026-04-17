import typing
from typing import Literal, Union, overload

import numpy as np
from numpy.typing import NDArray

_T = typing.TypeVar("_T", bound=np.generic)

class DimensionError(Exception): ...

@overload
def fill(
    labels: NDArray[_T],
    in_place: bool = False,
    *,
    return_fill_count: Literal[False] = False,
) -> NDArray[_T]: ...
@overload
def fill(
    labels: NDArray[_T],
    in_place: bool,
    return_fill_count: Literal[False] = False,
) -> NDArray[_T]: ...
@overload
def fill(
    labels: NDArray[_T],
    in_place: bool = False,
    *,
    return_fill_count: Literal[True],
) -> tuple[NDArray[_T], int]: ...
@overload
def fill(
    labels: NDArray[_T],
    in_place: bool,
    return_fill_count: Literal[True],
) -> tuple[NDArray[_T], int]: ...
def fill(  # type: ignore[misc]
    labels: NDArray[_T], in_place: bool = False, return_fill_count: bool = False
) -> Union[NDArray[_T], tuple[NDArray[_T], int]]:
    """Fills holes in a 1D, 2D, or 3D binary image.

    Args:
        labels: a binary valued numpy array of any common
            integer or floating dtype
        in_place: bool, Allow modification of the input array (saves memory)
        return_fill_count: Also return the number of voxels that were filled in.

    Returns:
        A void filled binary image of the same dtype as labels with the number
        of filled in background voxels if return_fill_count is True.
    """

def void_shard() -> None: ...
@overload
def fill_multi_label(
    labels: NDArray[_T],
    in_place: bool = False,
    *,
    return_fill_count: Literal[False] = False,
    connectivity: typing.Optional[int] = None,
) -> NDArray[_T]: ...
@overload
def fill_multi_label(
    labels: NDArray[_T],
    in_place: bool = False,
    *,
    return_fill_count: Literal[True],
    connectivity: typing.Optional[int] = None,
) -> tuple[NDArray[_T], int]: ...
def fill_multi_label(  # type: ignore[misc]
    labels: NDArray[_T],
    in_place: bool = False,
    return_fill_count: bool = False,
    connectivity: typing.Optional[int] = None,
) -> Union[NDArray[_T], tuple[NDArray[_T], int]]:
    """Multi-label generalization of :func:`fill`.

    For each maximal connected region of ``0`` voxels (a void), fills it
    with label ``L`` iff every path from the void to the image exterior
    passes through a voxel of ``L`` (``L`` is a RAG dominator of the
    void). Voids trapped between two distinct shells are left unfilled.
    """
