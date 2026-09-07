"""Fast Discrete Curvelet Transform via wedge wrapping.

A NumPy port of CurveLab's `fdct_wrapping.m` / `ifdct_wrapping.m` /
`fdct_wrapping_window.m` (Laurent Demanet, 2004), kept in
`hj_code/curvelet_interpolation/`.  The port is deliberately literal: index
arithmetic is carried in MATLAB's 1-based convention and converted to 0-based
only at the point of indexing, so the two can be read side by side.

Coefficient layout
------------------
MATLAB `C{j}{l}` maps to Python `C[j - 1][l - 1]`.  `C[0][0]` is the coarsest
wavelet level; `C[-1]` is the finest.  Each entry is a complex 2-D array
(real when `is_real=1`).

MATLAB constructs that needed care
----------------------------------
* `round` rounds half away from zero; `numpy.round` rounds half to even, so
  `mround` is used wherever the original called `round`.
* `A(B) = A(B) + C` with repeated indices in B keeps the *last* occurrence.
  NumPy's `A[B] += C` behaves the same way (gather, add, scatter), and the
  inverse transform depends on it - the corner wedges clamp their column
  indices, so duplicates are real.  Those scatters are done row by row to keep
  the ordering explicit rather than relying on NumPy's iteration order.
* Divisions by zero along the wedge centre lines produce inf/nan here exactly
  as they do in MATLAB; `fdct_wrapping_window` maps them to finite window
  values.  The warnings are silenced rather than the values changed.

The per-scale wedge geometry depends only on the image size and the transform
parameters, so it is built once and cached; the forward and inverse share it.
"""

import numpy as np

__all__ = ["fdct_wrapping_window", "fdct_wrapping", "ifdct_wrapping",
           "curvelet_nbangles"]


def mround(a):
    """MATLAB's round: halves go away from zero."""
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _fl(v):
    """floor as a Python int, for index arithmetic."""
    return int(np.floor(v))


def fdct_wrapping_window(x):
    """The two halves of a C-infinity compactly supported window.

    Port of fdct_wrapping_window.m.  Returns (wl, wr) in that order, matching
    the MATLAB output order.
    """
    x = np.array(x, dtype=float, copy=True)
    x[np.abs(x) < 2.0 ** -52] = 0.0

    wr = np.zeros(x.shape)
    wl = np.zeros(x.shape)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inner = (x > 0) & (x < 1)
        wr[inner] = np.exp(1 - 1.0 / (1 - np.exp(1 - 1.0 / x[inner])))
        wr[x <= 0] = 1.0
        wl[inner] = np.exp(1 - 1.0 / (1 - np.exp(1 - 1.0 / (1 - x[inner]))))
        wl[x >= 1] = 1.0

        normalization = np.sqrt(wl ** 2 + wr ** 2)
        wr = wr / normalization
        wl = wl / normalization
    return wl, wr


def curvelet_nbangles(nbscales, nbangles_coarse, finest):
    """Number of wedges per scale, coarsest first (MATLAB `nbangles`)."""
    k = np.arange(nbscales - 1)                       # nbscales - (nbscales:-1:2)
    nbangles = [1] + list((nbangles_coarse * 2 ** np.ceil(k / 2)).astype(int))
    if finest == 2:
        nbangles[nbscales - 1] = 1
    return nbangles


def _lowpass(M1, M2, trim_for_mod3=False, N1=None, N2=None):
    """Separable lowpass window, shared by every pyramid step."""
    wl1 = _fl(2 * M1) - _fl(M1) - 1
    wl2 = _fl(2 * M2) - _fl(M2) - 1
    if trim_for_mod3:
        wl1 -= (N1 % 3 == 0)
        wl2 -= (N2 % 3 == 0)

    wl_1, wr_1 = fdct_wrapping_window(np.linspace(0.0, 1.0, wl1 + 1))
    wl_2, wr_2 = fdct_wrapping_window(np.linspace(0.0, 1.0, wl2 + 1))

    lp1 = np.concatenate([wl_1, np.ones(2 * _fl(M1) + 1), wr_1])
    lp2 = np.concatenate([wl_2, np.ones(2 * _fl(M2) + 1), wr_2])
    if trim_for_mod3:
        if N1 % 3 == 0:
            lp1 = np.concatenate([[0.0], lp1, [0.0]])
        if N2 % 3 == 0:
            lp2 = np.concatenate([[0.0], lp2, [0.0]])
    return np.outer(lp1, lp2)


class _Wedge:
    """Geometry and window for one wedge, shared by forward and inverse.

    `rows`/`adm` are 1-based indices into the (rotated) working spectrum,
    `new_row` the 1-based destination rows in the wrapped array.  `extra` is
    the `(cols > 0)` / `(cols <= limit)` factor the corner wedges apply on the
    way in, or None.  `corner` is "left", "right" or None; the corner wedges
    clamp their column indices, so `adm` repeats values there and the inverse
    scatter has to stay ordered - and the two corners order it differently.
    """

    __slots__ = ("rows", "new_row", "adm", "extra", "window", "shape", "corner")

    def __init__(self, rows, new_row, adm, extra, window, corner=None):
        self.rows = rows
        self.new_row = new_row
        self.adm = adm
        self.extra = extra
        self.window = window
        self.shape = adm.shape
        self.corner = corner


def _wrap_indices(left_line, width_wedge, first_col, rows, first_row, period):
    """The `cols` / `new_row` bookkeeping shared by every wedge.

    `new_row` is a permutation of 1..period (row runs over a full period), so
    scattering into it never collides.
    """
    offs = np.arange(width_wedge)
    cols = left_line[:, None] + (offs[None, :]
                                 - (left_line[:, None] - first_col)) % width_wedge
    new_row = 1 + (rows - first_row) % period
    return cols, new_row


def _scale_geometry(M1, M2, nbangles_j, is_real):
    """Wedge geometry for one scale, as a list over quadrants of lists of
    _Wedge.  Mirrors the angular loop of fdct_wrapping.m, which the inverse
    repeats verbatim."""
    nbquadrants = 2 + 2 * (not is_real)
    nbangles_perquad = nbangles_j // 4
    quadrants = []

    for quadrant in range(1, nbquadrants + 1):
        odd = (quadrant % 2 == 1)
        M_horiz = M2 if odd else M1
        M_vert = M1 if odd else M2
        f4h = _fl(4 * M_horiz)
        f4v = _fl(4 * M_vert)

        ticks_left = mround(np.linspace(0.0, 0.5, nbangles_perquad + 1) * 2 * f4h + 1)
        ticks_right = 2 * f4h + 2 - ticks_left
        if nbangles_perquad % 2:
            wedge_ticks = np.concatenate([ticks_left, ticks_right[::-1]])
        else:
            wedge_ticks = np.concatenate([ticks_left, ticks_right[-2::-1]])
        we = wedge_ticks[1:-1:2]                      # wedge_endpoints
        wm = (we[:-1] + we[1:]) / 2.0                 # wedge_midpoints

        # MATLAB: quadrant-2 == mod(quadrant-2,2) is true for quadrants 2,3;
        # quadrant-3 == mod(quadrant-3,2) is true for quadrants 3,4.
        par_row = int((quadrant - 2) == ((quadrant - 2) % 2))
        par_col = int((quadrant - 3) == ((quadrant - 3) % 2))

        wedges = []

        # ---- left corner wedge -------------------------------------------
        first_wedge_endpoint_vert = int(mround(2 * f4v / (2 * nbangles_perquad) + 1))
        lcw = f4v - _fl(M_vert) + int(np.ceil(first_wedge_endpoint_vert / 4))
        Y_corner = np.arange(1, lcw + 1)
        XX, YY = np.meshgrid(np.arange(1, 2 * f4h + 2), Y_corner)

        width_wedge = int(we[1] + we[0] - 1)
        slope_wedge = (f4h + 1 - we[0]) / f4v
        left_line = mround(2 - we[0] + slope_wedge * (Y_corner - 1)).astype(int)
        first_row = f4v + 2 - int(np.ceil((lcw + 1) / 2)) + ((lcw + 1) % 2) * par_row
        first_col = f4h + 2 - int(np.ceil((width_wedge + 1) / 2)) \
            + ((width_wedge + 1) % 2) * par_col

        cols, new_row = _wrap_indices(left_line, width_wedge, first_col,
                                      Y_corner, first_row, lcw)
        adm = mround(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)   # max(cols, 1)

        wXX = np.zeros((lcw, width_wedge))
        wYY = np.zeros((lcw, width_wedge))
        wXX[new_row - 1, :] = XX[Y_corner[:, None] - 1, adm - 1]
        wYY[new_row - 1, :] = YY[Y_corner[:, None] - 1, adm - 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            slope_right = (f4h + 1 - wm[0]) / f4v
            mid_right = wm[0] + slope_right * (wYY - 1)
            coord_right = 0.5 + f4v / (we[1] - we[0]) * (wXX - mid_right) / (f4v + 1 - wYY)

            C2 = 1.0 / (1.0 / (2 * f4h / (we[0] - 1) - 1)
                        + 1.0 / (2 * f4v / (first_wedge_endpoint_vert - 1) - 1))
            C1 = C2 / (2 * f4v / (first_wedge_endpoint_vert - 1) - 1)
            on_edge = ((wXX - 1) / f4h + (wYY - 1) / f4v) == 2
            wXX_c = wXX.copy()
            wXX_c[on_edge] += 1
            coord_corner = C1 + C2 * ((wXX_c - 1) / f4h - (wYY - 1) / f4v) / \
                (2 - ((wXX_c - 1) / f4h + (wYY - 1) / f4v))

        wl_left, _ = fdct_wrapping_window(coord_corner)
        _, wr_right = fdct_wrapping_window(coord_right)
        wedges.append(_Wedge(Y_corner, new_row, adm, (cols > 0).astype(float),
                             wl_left * wr_right, corner="left"))

        # ---- regular wedges ----------------------------------------------
        length_wedge = f4v - _fl(M_vert)
        Y = np.arange(1, length_wedge + 1)
        first_row_reg = f4v + 2 - int(np.ceil((length_wedge + 1) / 2)) \
            + ((length_wedge + 1) % 2) * par_row

        for subl in range(2, nbangles_perquad):       # MATLAB 2:(nbangles_perquad-1)
            i = subl - 1                              # 0-based into we / wm
            width_wedge = int(we[i + 1] - we[i - 1] + 1)
            slope_wedge = ((f4h + 1) - we[i]) / f4v
            left_line = mround(we[i - 1] + slope_wedge * (Y - 1)).astype(int)
            first_col = f4h + 2 - int(np.ceil((width_wedge + 1) / 2)) \
                + ((width_wedge + 1) % 2) * par_col

            cols, new_row = _wrap_indices(left_line, width_wedge, first_col,
                                          Y, first_row_reg, length_wedge)
            wXX = np.zeros((length_wedge, width_wedge))
            wYY = np.zeros((length_wedge, width_wedge))
            wXX[new_row - 1, :] = XX[Y[:, None] - 1, cols - 1]
            wYY[new_row - 1, :] = YY[Y[:, None] - 1, cols - 1]

            with np.errstate(divide="ignore", invalid="ignore"):
                slope_left = ((f4h + 1) - wm[i - 1]) / f4v
                mid_left = wm[i - 1] + slope_left * (wYY - 1)
                coord_left = 0.5 + f4v / (we[i] - we[i - 1]) * \
                    (wXX - mid_left) / (f4v + 1 - wYY)
                slope_right = ((f4h + 1) - wm[i]) / f4v
                mid_right = wm[i] + slope_right * (wYY - 1)
                coord_right = 0.5 + f4v / (we[i + 1] - we[i]) * \
                    (wXX - mid_right) / (f4v + 1 - wYY)

            wl_left, _ = fdct_wrapping_window(coord_left)
            _, wr_right = fdct_wrapping_window(coord_right)
            wedges.append(_Wedge(Y, new_row, cols, None,
                                 wl_left * wr_right))

        # ---- right corner wedge ------------------------------------------
        width_wedge = int(4 * f4h + 3 - we[-1] - we[-2])
        slope_wedge = ((f4h + 1) - we[-1]) / f4v
        left_line = mround(we[-2] + slope_wedge * (Y_corner - 1)).astype(int)
        first_row = f4v + 2 - int(np.ceil((lcw + 1) / 2)) + ((lcw + 1) % 2) * par_row
        first_col = f4h + 2 - int(np.ceil((width_wedge + 1) / 2)) \
            + ((width_wedge + 1) % 2) * par_col

        cols, new_row = _wrap_indices(left_line, width_wedge, first_col,
                                      Y_corner, first_row, lcw)
        limit = 2 * f4h + 1
        adm = mround(0.5 * (cols + limit - np.abs(cols - limit))).astype(int)  # min(cols, limit)

        wXX = np.zeros((lcw, width_wedge))
        wYY = np.zeros((lcw, width_wedge))
        wXX[new_row - 1, :] = XX[Y_corner[:, None] - 1, adm - 1]
        wYY[new_row - 1, :] = YY[Y_corner[:, None] - 1, adm - 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            slope_left = ((f4h + 1) - wm[-1]) / f4v
            mid_left = wm[-1] + slope_left * (wYY - 1)
            coord_left = 0.5 + f4v / (we[-1] - we[-2]) * (wXX - mid_left) / (f4v + 1 - wYY)

            C2 = -1.0 / (2 * f4h / (we[-1] - 1) - 1
                         + 1.0 / (2 * f4v / (first_wedge_endpoint_vert - 1) - 1))
            C1 = -C2 * (2 * f4h / (we[-1] - 1) - 1)
            on_edge = ((wXX - 1) / f4h) == ((wYY - 1) / f4v)
            wXX_c = wXX.copy()
            wXX_c[on_edge] -= 1
            coord_corner = C1 + C2 * (2 - ((wXX_c - 1) / f4h + (wYY - 1) / f4v)) / \
                ((wXX_c - 1) / f4h - (wYY - 1) / f4v)

        wl_left, _ = fdct_wrapping_window(coord_left)
        _, wr_right = fdct_wrapping_window(coord_corner)
        wedges.append(_Wedge(Y_corner, new_row, adm, (cols <= limit).astype(float),
                             wl_left * wr_right, corner="right"))

        quadrants.append(wedges)

    return quadrants


_GEOM_CACHE = {}


def _geometry(M1, M2, nbangles_j, is_real):
    key = (round(M1, 9), round(M2, 9), nbangles_j, bool(is_real))
    if key not in _GEOM_CACHE:
        _GEOM_CACHE[key] = _scale_geometry(M1, M2, nbangles_j, is_real)
    return _GEOM_CACHE[key]


def _fft2c(a):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a))) / np.sqrt(a.size)


def _ifft2c(a):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a))) * np.sqrt(a.size)


def fdct_wrapping(x, is_real=0, finest=2, nbscales=None, nbangles_coarse=16):
    """Forward transform.  Port of fdct_wrapping.m.

    Returns a list of lists: `C[j][l]`, coarsest scale first.
    """
    x = np.asarray(x)
    X = _fft2c(x)
    N1, N2 = X.shape
    if nbscales is None:
        nbscales = int(np.ceil(np.log2(min(N1, N2)) - 3))

    nbangles = curvelet_nbangles(nbscales, nbangles_coarse, finest)
    C = [[None] * nbangles[j] for j in range(nbscales)]

    M1 = N1 / 3.0
    M2 = N2 / 3.0

    if finest == 1:
        bigN1 = 2 * _fl(2 * M1) + 1
        bigN2 = 2 * _fl(2 * M2) + 1
        eq1 = (_fl(N1 / 2) - _fl(2 * M1) + np.arange(bigN1)) % N1
        eq2 = (_fl(N2 / 2) - _fl(2 * M2) + np.arange(bigN2)) % N2
        X = X[np.ix_(eq1, eq2)]
        Xlow = X * _lowpass(M1, M2, trim_for_mod3=True, N1=N1, N2=N2)
        scales = list(range(nbscales, 1, -1))
    else:
        M1 /= 2.0
        M2 /= 2.0
        lowpass = _lowpass(M1, M2)
        hipass = np.sqrt(1 - lowpass ** 2)
        i1 = np.arange(-_fl(2 * M1), _fl(2 * M1) + 1) + int(np.ceil((N1 + 1) / 2)) - 1
        i2 = np.arange(-_fl(2 * M2), _fl(2 * M2) + 1) + int(np.ceil((N2 + 1) / 2)) - 1
        Xlow = X[np.ix_(i1, i2)] * lowpass
        Xhi = X.copy()
        Xhi[np.ix_(i1, i2)] *= hipass
        fine = _ifft2c(Xhi)
        C[nbscales - 1][0] = np.real(fine) if is_real else fine
        scales = list(range(nbscales - 1, 1, -1))

    for j in scales:
        M1 /= 2.0
        M2 /= 2.0
        lowpass = _lowpass(M1, M2)
        hipass = np.sqrt(1 - lowpass ** 2)

        Xhi = Xlow.copy()                 # size 2*floor(4*M1)+1 x 2*floor(4*M2)+1
        i1 = np.arange(-_fl(2 * M1), _fl(2 * M1) + 1) + _fl(4 * M1)
        i2 = np.arange(-_fl(2 * M2), _fl(2 * M2) + 1) + _fl(4 * M2)
        Xlow = Xhi[np.ix_(i1, i2)]
        Xhi[np.ix_(i1, i2)] = Xlow * hipass
        Xlow = Xlow * lowpass

        l = 0
        for q, wedges in enumerate(_geometry(M1, M2, nbangles[j - 1], is_real)):
            for wd in wedges:
                wrapped = np.zeros(wd.shape, dtype=complex)
                gathered = Xhi[wd.rows[:, None] - 1, wd.adm - 1]
                if wd.extra is not None:
                    gathered = gathered * wd.extra
                wrapped[wd.new_row - 1, :] = gathered
                wrapped = wrapped * wd.window
                wrapped = np.rot90(wrapped, -q)
                coef = _ifft2c(wrapped)

                l += 1
                if is_real:
                    C[j - 1][l - 1] = np.sqrt(2) * np.real(coef)
                    C[j - 1][l - 1 + nbangles[j - 1] // 2] = np.sqrt(2) * np.imag(coef)
                else:
                    C[j - 1][l - 1] = coef

            if q + 1 < len(_geometry(M1, M2, nbangles[j - 1], is_real)):
                Xhi = np.rot90(Xhi)

    C[0][0] = np.real(_ifft2c(Xlow)) if is_real else _ifft2c(Xlow)
    return C


def ifdct_wrapping(C, is_real=0, M=None, N=None):
    """Inverse (adjoint, also pseudo-inverse) transform.  Port of
    ifdct_wrapping.m."""
    nbscales = len(C)
    nbangles_coarse = len(C[1])
    finest = 2 if len(C[-1]) == 1 else 1
    nbangles = curvelet_nbangles(nbscales, nbangles_coarse, finest)

    if M is None or N is None:
        if finest == 1:
            raise ValueError("ifdct_wrapping(C, is_real, M, N) needs the output "
                             "size when finest == 1")
        N1, N2 = C[-1][0].shape
    else:
        N1, N2 = M, N

    M1 = N1 / 3.0
    M2 = N2 / 3.0

    if finest == 1:
        bigN1 = 2 * _fl(2 * M1) + 1
        bigN2 = 2 * _fl(2 * M2) + 1
        X = np.zeros((bigN1, bigN2), dtype=complex)
        lowpass = _lowpass(M1, M2, trim_for_mod3=True, N1=N1, N2=N2)
        hipass_finest = None
        scales = list(range(nbscales, 1, -1))
    else:
        M1 /= 2.0
        M2 /= 2.0
        bigN1 = 2 * _fl(2 * M1) + 1
        bigN2 = 2 * _fl(2 * M2) + 1
        X = np.zeros((bigN1, bigN2), dtype=complex)
        lowpass = _lowpass(M1, M2)
        hipass_finest = np.sqrt(1 - lowpass ** 2)
        scales = list(range(nbscales - 1, 1, -1))

    Xj_topleft_1 = 1
    Xj_topleft_2 = 1
    for j in scales:
        M1 /= 2.0
        M2 /= 2.0
        lowpass_next = _lowpass(M1, M2)
        hipass = np.sqrt(1 - lowpass_next ** 2)
        Xj = np.zeros((2 * _fl(4 * M1) + 1, 2 * _fl(4 * M2) + 1), dtype=complex)

        l = 0
        for q, wedges in enumerate(_geometry(M1, M2, nbangles[j - 1], is_real)):
            for wd in wedges:
                l += 1
                if is_real:
                    z = C[j - 1][l - 1] + 1j * C[j - 1][l - 1 + nbangles[j - 1] // 2]
                    wrapped = _fft2c(z) / np.sqrt(2)
                else:
                    wrapped = _fft2c(C[j - 1][l - 1])
                wrapped = np.rot90(wrapped, q)
                wrapped = wrapped * wd.window

                if wd.corner is None:
                    # cols are a permutation within each row: no duplicates,
                    # so the whole wedge can go in one scatter.
                    Xj[wd.rows[:, None] - 1, wd.adm - 1] += wrapped[wd.new_row - 1, :]
                elif wd.corner == "right":
                    # MATLAB writes Xj(row, fliplr(adm)) += wrapped(:, end:-1:1),
                    # reversing the column order so that among the clamped
                    # duplicates the leftmost one ends up surviving.
                    for r in range(len(wd.rows)):
                        cols = wd.adm[r][::-1] - 1
                        Xj[wd.rows[r] - 1, cols] = \
                            Xj[wd.rows[r] - 1, cols] + wrapped[wd.new_row[r] - 1, ::-1]
                else:
                    # Left corner: straight order, last occurrence survives.
                    for r in range(len(wd.rows)):
                        cols = wd.adm[r] - 1
                        Xj[wd.rows[r] - 1, cols] = \
                            Xj[wd.rows[r] - 1, cols] + wrapped[wd.new_row[r] - 1, :]

            Xj = np.rot90(Xj)

        Xj = Xj * lowpass
        k1 = np.arange(-_fl(2 * M1), _fl(2 * M1) + 1) + _fl(4 * M1)
        k2 = np.arange(-_fl(2 * M2), _fl(2 * M2) + 1) + _fl(4 * M2)
        Xj[np.ix_(k1, k2)] *= hipass

        loc1 = Xj_topleft_1 - 1 + np.arange(2 * _fl(4 * M1) + 1)
        loc2 = Xj_topleft_2 - 1 + np.arange(2 * _fl(4 * M2) + 1)
        X[np.ix_(loc1, loc2)] += Xj

        Xj_topleft_1 += _fl(4 * M1) - _fl(2 * M1)
        Xj_topleft_2 += _fl(4 * M2) - _fl(2 * M2)
        lowpass = lowpass_next

    if is_real:
        Y = X
        X = np.rot90(X, 2)
        X = X + np.conj(Y)

    # Coarsest wavelet level
    M1 /= 2.0
    M2 /= 2.0
    Xj = _fft2c(C[0][0])
    loc1 = Xj_topleft_1 - 1 + np.arange(2 * _fl(4 * M1) + 1)
    loc2 = Xj_topleft_2 - 1 + np.arange(2 * _fl(4 * M2) + 1)
    X[np.ix_(loc1, loc2)] += Xj * lowpass

    # Finest level
    M1 = N1 / 3.0
    M2 = N2 / 3.0
    if finest == 1:
        shift_1 = _fl(2 * M1) - _fl(N1 / 2)
        shift_2 = _fl(2 * M2) - _fl(N2 / 2)
        Y = X[:, shift_2:shift_2 + N2].copy()
        Y[:, N2 - shift_2:N2] += X[:, 0:shift_2]
        Y[:, 0:shift_2] += X[:, N2 + shift_2:N2 + 2 * shift_2]
        X = Y[shift_1:shift_1 + N1, :].copy()
        X[N1 - shift_1:N1, :] += Y[0:shift_1, :]
        X[0:shift_1, :] += Y[N1 + shift_1:N1 + 2 * shift_1, :]
    else:
        Y = _fft2c(C[nbscales - 1][0])
        t1 = int(np.ceil((N1 + 1) / 2)) - _fl(M1) - 1
        t2 = int(np.ceil((N2 + 1) / 2)) - _fl(M2) - 1
        loc1 = t1 + np.arange(2 * _fl(M1) + 1)
        loc2 = t2 + np.arange(2 * _fl(M2) + 1)
        Y[np.ix_(loc1, loc2)] = Y[np.ix_(loc1, loc2)] * hipass_finest + X
        X = Y

    x = _ifft2c(X)
    return np.real(x) if is_real else x
