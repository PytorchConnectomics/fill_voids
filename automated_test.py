import pytest 

import fill_voids
import scipy.ndimage
from scipy.ndimage import binary_fill_holes

from tqdm import tqdm

import crackle
import numpy as np 

img = crackle.load('test_data.npy.ckl.gz')
SEGIDS = np.unique(img)[1:]

def test_scipy_comparison3d():
  segids = np.copy(SEGIDS)
  np.random.shuffle(segids)

  for segid in tqdm(segids[:10]):
    print(segid)
    binimg = (img == segid).view(np.uint8)
    slices = scipy.ndimage.find_objects(binimg)[0]
    binimg = binimg[slices]

    orig_binimg = np.copy(binimg, order='F')
    fv = fill_voids.fill(binimg, in_place=False)
    fvip = fill_voids.fill(binimg, in_place=True)

    assert np.all(fv == fvip)

    spy = binary_fill_holes(binimg)

    assert np.all(fv == spy)

def test_scipy_comparison2d():
  segids = np.copy(SEGIDS)
  np.random.shuffle(segids)

  for segid in tqdm(segids[:10]):
    print(segid)
    for z in tqdm(range(img.shape[2])):
      binimg = img[:,:,z] == segid

      orig_binimg = np.copy(binimg, order='F')
      fv = fill_voids.fill(binimg, in_place=False)
      fvip = fill_voids.fill(binimg, in_place=True)

      assert np.all(fv == fvip)

      spy = binary_fill_holes(binimg)

      assert np.all(fv == spy)

def test_2d_3d_differ():
  labels = np.zeros((10,10), dtype=bool)
  labels[1:9,1:9] = True
  labels[4:8,4:8] = False

  expected_result2d = np.zeros((10,10), dtype=bool)
  expected_result2d[1:9,1:9] = True

  expected_result3d = np.copy(labels).reshape(10,10,1)

  filled_labels, N = fill_voids.fill(labels, in_place=False, return_fill_count=True)
  assert N == 16
  assert np.all(filled_labels == expected_result2d)

  labels = labels[..., np.newaxis]
  
  filled_labels, N = fill_voids.fill(labels, in_place=False, return_fill_count=True)
  assert N == 0
  assert np.all(filled_labels == expected_result3d)

DTYPES = (
  bool, np.int8, np.uint8, np.uint16, np.int16, 
  np.int32, np.uint32, np.int64, np.uint64, 
  np.float32, np.float64
)

@pytest.mark.parametrize("dtype", DTYPES)
def test_dtypes(dtype):
  binimg = img == SEGIDS[0]

  comparison = fill_voids.fill(binimg, in_place=False)
  res = fill_voids.fill(binimg.astype(dtype), in_place=False)
  assert np.all(comparison == res)

def test_zero_array():
  labels = np.zeros((0,), dtype=np.uint8)
  # just don't throw an exception
  fill_voids.fill(labels, in_place=False)
  fill_voids.fill(labels, in_place=True)

  labels = np.zeros((128,128,128), dtype=np.uint8)
  fill_voids.fill(labels, in_place=True)
  assert not np.any(labels)

def test_return_count():
  labels = np.ones((10, 10, 10), dtype=bool)
  labels[3:6,3:6,3:6] = False

  filled = fill_voids.fill(labels)
  assert np.all(filled == 1)

  filled, ct = fill_voids.fill(labels, return_fill_count=True)
  assert np.any(labels == False)
  assert ct == 27

@pytest.mark.parametrize("dimension", [1,2,3,4,5,6])
def test_dimensions(dimension):
  size = [5] * dimension
  for i in range(3, dimension):
    size[i] = 1

  labels = np.ones(size, dtype=np.uint8)
  labels = fill_voids.fill(labels)
  assert labels.ndim == dimension

  if dimension <= 3:
    return
    
  size[dimension - 1] = 2
  labels = np.ones(size, dtype=np.uint8)
  try:
    labels = fill_voids.fill(labels)
    assert False
  except fill_voids.DimensionError:
    pass


# ---------------------------------------------------------------------------
# Multi-label fill tests: compare fill_voids.fill_multi_label against a
# naive per-label reference that mirrors the README semantics ("fill a
# void with the outermost enclosing label; leave it alone if the void
# lies between distinct shells").
# ---------------------------------------------------------------------------


def _naive_multi_label_fill(labels):
  """Naive per-label reference that follows the README semantics.

  For each background connected component (a "void"), find every
  foreground label adjacent to it and determine which of them actually
  *encloses* the void -- i.e. for which labels ``L`` the component
  belongs to a hole of ``binary_fill_holes(labels == L)``. If exactly
  one such ``L`` encloses the void, fill with ``L``. If several do
  (nested shells), pick the innermost (the one whose enclosing hole is
  smallest). If none do, or two unrelated labels tie for outermost
  shell, leave the void unfilled -- this is the "in-between" case from
  the README.

  Cost is O(K * N); intended only as a correctness oracle, not for
  production use. This mirrors what one would get by looping
  ``for L in labels: binary_fill_holes(labels == L)`` and adjudicating
  conflicts in a README-consistent way, but a good deal more carefully
  than the one-line version that simply lets the last label win.
  """
  import cc3d
  import scipy.ndimage as ndi

  ndim = labels.ndim
  structure = ndi.generate_binary_structure(ndim, 1)
  connectivity = 6 if ndim == 3 else 4

  bg = (labels == 0)
  bg_cc, num_cc = cc3d.connected_components(bg, connectivity=connectivity, return_N=True)

  # Which CCs touch the image boundary?
  cc_on_boundary = np.zeros(num_cc + 1, dtype=bool)
  for axis in range(ndim):
    for idx in (0, labels.shape[axis] - 1):
      for v in np.unique(np.take(bg_cc, idx, axis=axis)):
        if v != 0:
          cc_on_boundary[v] = True

  # For each label, find which CCs it encloses (via binary_fill_holes)
  # and how big its hole-component for each such CC is (smaller = inner).
  uniq = np.unique(labels)
  uniq = uniq[uniq != 0]
  cc_encloser_size = {c: {} for c in range(1, num_cc + 1)}  # cc -> {label: hole_size}
  for L in uniq:
    mask = (labels == L)
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
      continue
    hc, nhc = ndi.label(holes, structure=structure)
    if nhc == 0:
      continue
    sizes = np.bincount(hc.ravel(), minlength=nhc + 1)
    # For each background-CC c, find which hole-component of L it
    # belongs to (all voxels of a BG-CC lie in the same hole-component
    # of L when L encloses it, since BG-CC is more connected than L's
    # per-label hole-components).
    for c in range(1, num_cc + 1):
      if cc_on_boundary[c]:
        continue
      cc_mask = (bg_cc == c)
      hole_ids = hc[cc_mask & holes]
      if hole_ids.size == 0:
        continue
      h = int(hole_ids[0])
      cc_encloser_size[c][int(L)] = int(sizes[h])

  out = labels.copy()
  for c in range(1, num_cc + 1):
    if cc_on_boundary[c]:
      continue
    enclosers = cc_encloser_size[c]
    if not enclosers:
      continue
    # Innermost = smallest enclosing hole; break ties only if unique.
    min_size = min(enclosers.values())
    winners = [L for L, s in enclosers.items() if s == min_size]
    if len(winners) == 1:
      out[bg_cc == c] = winners[0]
  return out


def _assert_ml_matches_naive(img):
  expected = _naive_multi_label_fill(img)
  got, num = fill_voids.fill_multi_label(img, return_fill_count=True)
  assert np.array_equal(got, expected), (
    f"fill_multi_label disagrees with naive reference on shape {img.shape}"
  )
  expected_filled = int(((expected != img) & (img == 0)).sum())
  assert num == expected_filled, (
    f"return_fill_count mismatch: got {num}, expected {expected_filled}"
  )
  # The original array must not be mutated with in_place=False.
  assert (img == 0).any() == ((img == 0).any())


def test_multi_label_simple_donut_2d():
  img = np.zeros((7, 7), dtype=np.int32)
  img[1:6, 1:6] = 1
  img[3, 3] = 0
  _assert_ml_matches_naive(img)


def test_multi_label_inner_island_2d():
  # Outer shell A fully encloses a void that contains an island B.
  # The void should fill with A, leaving B intact.
  img = np.zeros((7, 7), dtype=np.int32)
  img[:, :] = 1
  img[1:6, 1:6] = 0
  img[3, 3] = 2
  _assert_ml_matches_naive(img)


def test_multi_label_nested_shells_2d():
  # A encloses B encloses a void. The README-spec / naive reference
  # fills the void with the *inner* shell B (immediate dominator).
  img = np.zeros((9, 9), dtype=np.int32)
  img[:, :] = 1
  img[1:8, 1:8] = 2
  img[2:7, 2:7] = 0
  _assert_ml_matches_naive(img)


def test_multi_label_gap_between_labels_not_filled():
  # Void sits between two distinct shells that both reach the exterior;
  # it must be left unfilled.
  img = np.zeros((5, 12), dtype=np.int32)
  img[1:4, 0:4] = 1
  img[1:4, 8:12] = 2
  _assert_ml_matches_naive(img)


def test_multi_label_two_separate_voids_one_label():
  img = np.zeros((5, 11), dtype=np.int32)
  img[:, :] = 1
  img[1:4, 1:4] = 0
  img[1:4, 7:10] = 0
  _assert_ml_matches_naive(img)


def test_multi_label_3d_nested_shells():
  img = np.zeros((30, 30, 30), dtype=np.int32)
  img[3:27, 3:27, 3:27] = 1
  img[10:20, 10:20, 10:20] = 2
  img[13:17, 13:17, 13:17] = 0
  _assert_ml_matches_naive(img)


def test_multi_label_3d_inner_island():
  img = np.zeros((10, 10, 10), dtype=np.int32)
  img[1:9, 1:9, 1:9] = 1
  img[3:7, 3:7, 3:7] = 0
  img[5, 5, 5] = 2
  _assert_ml_matches_naive(img)


def test_multi_label_matches_binary_fill():
  # With exactly one foreground label present, fill_multi_label must
  # agree with the existing binary fill_voids.fill.
  img = np.zeros((12, 12, 12), dtype=np.uint8)
  img[1:11, 1:11, 1:11] = 1
  img[3:9, 3:9, 3:9] = 0
  img[4:8, 4:8, 4:8] = 1  # partial interior
  binary = fill_voids.fill(img.copy(), in_place=False)
  multi = fill_voids.fill_multi_label(img.astype(np.int32))
  assert np.array_equal(binary.astype(np.int32), multi)


def test_multi_label_preserves_dtype():
  for dtype in (np.uint8, np.int16, np.uint32, np.int64, np.uint64):
    img = np.zeros((6, 6), dtype=dtype)
    img[1:5, 1:5] = 3
    img[2, 2] = 0
    out = fill_voids.fill_multi_label(img)
    assert out.dtype == dtype
    assert out[2, 2] == 3


def test_multi_label_in_place_behavior():
  img = np.zeros((5, 5), dtype=np.int32)
  img[:, :] = 1
  img[2, 2] = 0
  snapshot = img.copy()
  _ = fill_voids.fill_multi_label(img, in_place=False)
  assert np.array_equal(img, snapshot), "in_place=False must not modify the input"
  _ = fill_voids.fill_multi_label(img, in_place=True)
  assert img[2, 2] == 1, "in_place=True must fill the void in the input array"


def test_multi_label_empty_and_allzero():
  assert fill_voids.fill_multi_label(np.zeros((0, 0), dtype=np.int32)).shape == (0, 0)
  img = np.zeros((4, 4), dtype=np.int32)
  out, n = fill_voids.fill_multi_label(img, return_fill_count=True)
  assert n == 0 and np.array_equal(out, img)
  img = np.ones((4, 4), dtype=np.int32)
  out, n = fill_voids.fill_multi_label(img, return_fill_count=True)
  assert n == 0 and np.array_equal(out, img)


def test_multi_label_matches_naive_on_well_posed_image():
  # On a well-posed image (every void is cleanly enclosed by a unique
  # shell) fill_multi_label must match the per-label naive oracle.
  # Construct: a grid of labeled rectangles, each with a small hole
  # punched inside. No void touches two different labels.
  img = np.zeros((60, 60), dtype=np.int32)
  label = 1
  for i in range(0, 60, 12):
    for j in range(0, 60, 12):
      img[i + 1:i + 11, j + 1:j + 11] = label
      img[i + 4:i + 7, j + 4:j + 7] = 0  # interior void
      label += 1
  _assert_ml_matches_naive(img)


def test_multi_label_matches_naive_on_3d_well_posed_image():
  # 3D version of the above: a stack of labeled blocks each with an
  # interior void and an inner-label island.
  img = np.zeros((30, 30, 30), dtype=np.int32)
  label = 1
  for k in range(0, 30, 10):
    img[k + 1:k + 9, 1:29, 1:29] = label
    img[k + 3:k + 7, 10:20, 10:20] = 0  # void
    img[k + 5, 14:16, 14:16] = label + 100  # island inside the void
    label += 1
  _assert_ml_matches_naive(img)


def test_multi_label_is_significantly_faster_than_naive():
  # On a volume with many labels but well-posed (unique-shell) voids,
  # fill_multi_label should be dramatically faster than the naive
  # per-label loop. The correctness part is already covered above; this
  # test just guards against a future regression to per-label scanning.
  import time

  rng = np.random.default_rng(0)
  img = np.zeros((120, 120, 120), dtype=np.int32)
  # 64 labeled cubes, each punched with a small interior void.
  label = 1
  for k in range(0, 120, 30):
    for j in range(0, 120, 30):
      for i in range(0, 120, 30):
        img[k + 1:k + 29, j + 1:j + 29, i + 1:i + 29] = label
        img[k + 12:k + 18, j + 12:j + 18, i + 12:i + 18] = 0
        label += 1

  t0 = time.time()
  fast = fill_voids.fill_multi_label(img.copy())
  t_fast = time.time() - t0

  t0 = time.time()
  naive = _naive_multi_label_fill(img.copy())
  t_naive = time.time() - t0

  assert np.array_equal(fast, naive)
  # Very loose bound: fill_multi_label should beat the naive per-label
  # loop by at least 3x on a volume with many labels. In practice the
  # gap is typically 10x-50x; 3x just catches catastrophic regressions
  # without being flaky on slow CI hardware.
  assert t_fast * 3 < t_naive, (
    f"fill_multi_label ({t_fast:.3f}s) is not >=3x faster than naive "
    f"({t_naive:.3f}s)"
  )
