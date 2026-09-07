import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import os
from utils.data import get_project_root
import pickle


def hilbert_1d(data):
    nt, nr = data.shape

    data_fft = np.fft.fftn(data, axes=(-2, -1))

    freqs_t = np.fft.fftfreq(nt)

    hilbert_filter_t = np.where(freqs_t > 0, 1j, np.where(freqs_t < 0, -1j, 0))[:, None]

    data_fft = hilbert_filter_t * data_fft

    data_hilbert = np.fft.ifftn(data_fft, axes=(-2, -1))
    return data_hilbert


def envelope_1d(data):
    data_hilbert = hilbert_1d(data)
    data_env = np.real(np.sqrt(data_hilbert * data_hilbert + data * data))
    data_env = data_env.astype(np.float32)
    return data_env


def hilbert_2d(data):
    nx, ny = data.shape

    data_fft = np.fft.fft2(data)

    freqs_x = np.fft.fftfreq(nx)
    freqs_y = np.fft.fftfreq(ny)

    epsilon = 1e-8
    kx_grid, ky_grid = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    k_magnitude = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid + epsilon)
    is_dc = k_magnitude < 2 * epsilon

    H_riesz_x = np.zeros_like(kx_grid, dtype=np.complex64)
    H_riesz_y = np.zeros_like(ky_grid, dtype=np.complex64)

    H_riesz_x[~is_dc] = -1j * kx_grid[~is_dc] / k_magnitude[~is_dc]
    H_riesz_y[~is_dc] = -1j * ky_grid[~is_dc] / k_magnitude[~is_dc]

    F_Rx_data = data_fft * H_riesz_x
    F_Ry_data = data_fft * H_riesz_y

    Rx_data = np.fft.ifft2(F_Rx_data)
    Ry_data = np.fft.ifft2(F_Ry_data)

    return Rx_data, Ry_data


def envelope_2d(data):
    Rt_data, Rr_data = hilbert_2d(data)
    data_env = np.real(np.sqrt(Rt_data * Rt_data + Rr_data * Rr_data + data * data))
    data_env = data_env.astype(np.float32)
    return data_env


def calculate_logscale(data, log_base=1, smooth_sigma=5.0, eps=1e-10, is_1d_envelope=True):
    data_env = envelope_1d(data) if is_1d_envelope else envelope_2d(data)
    data_env += log_base

    data_env_log = np.log10(data_env)
    data_env_log -= data_env_log.min()
    data_env_log += eps

    scale = data_env_log / data_env
    scale = np.log10(scale)
    scale = ndimage.gaussian_filter(scale, sigma=smooth_sigma)
    scale = np.power(10, scale)
    return scale


def calculate_norscale_inversion(data_log, log_base=1, smooth_sigma=5.0, eps=1e-10, iterations=50, is_1d_envelope=True):
    data_est = np.copy(data_log)
    est_scale = None

    for i in range(iterations):
        est_scale = calculate_logscale(data_est, log_base, smooth_sigma, eps, is_1d_envelope)
        data_est = data_log / est_scale

    return est_scale, data_est


def get_idx(iy, ix, shape):
    if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
        return shape[1] * iy + ix
    else:
        raise Exception


_in_memory_solvers_cache = {}
FACTORIZATION_CACHE_DIR = os.path.join(get_project_root(), "factorization_cache")
def _create_solver_from_components(L_mat, U_mat, perm_r_vec, perm_c_vec):
    def solve(b_vector):
        if b_vector.ndim > 1:
            b_vector = b_vector.ravel()

        b_rhs_permuted = b_vector[perm_r_vec]

        y1 = linalg.spsolve_triangular(L_mat, b_rhs_permuted, lower=True, unit_diagonal=False)
        y2 = linalg.spsolve_triangular(U_mat, y1, lower=False, unit_diagonal=False)

        x_final = np.zeros_like(y2, dtype=y2.dtype)
        x_final[perm_c_vec] = y2
        return x_final
    return solve


def mirror_padding(signal, pad_length=1000, taper_length=500, axis=-2):
    zero_length = pad_length - taper_length

    mirror = np.flip(signal, axis=axis)

    taper = np.hanning(2 * taper_length)
    shape = [1] * signal.ndim
    shape[axis] = 2 * taper_length
    taper = taper.reshape(shape)

    mirror_tapered_top = mirror[-taper_length:] * taper[:taper_length]
    mirror_tapered_bot = mirror[:taper_length] * taper[taper_length:]

    # zero padding
    shape_zero = list(signal.shape)
    shape_zero[axis] = zero_length
    zeros = np.zeros(shape_zero, dtype=signal.dtype)

    # concatenate along the axis
    padded = np.concatenate([zeros, mirror_tapered_top, signal, mirror_tapered_bot, zeros], axis=axis)
    return padded


def smooth_padding(data, alp=10, lmb_b=0.5, sigma=2.0, perc=1.0):
    data_gau = ndimage.gaussian_filter(data, sigma)
    shape = data.shape
    scale = np.quantile(np.abs(data), perc) / np.max(np.abs(data))

    row = []
    col = []
    val = []
    for iy in range(shape[0]):
        for ix in range(shape[1]):
            val_dig = 0

            if iy != 0:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy - 1, ix, shape))
                val.append(1.0)
                val_dig += 1.0

            if iy != shape[0] - 1:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy + 1, ix, shape))
                val.append(1.0)
                val_dig += 1.0

            if ix != 0:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy, ix - 1, shape))
                val.append(1.0)
                val_dig += 1.0

            if ix != shape[1] - 1:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy, ix + 1, shape))
                val.append(1.0)
                val_dig += 1.0

            row.append(get_idx(iy, ix, shape))
            col.append(get_idx(iy, ix, shape))
            val.append(-val_dig)
    L_mat = sparse.csr_matrix((val, (row, col)), shape=(shape[0] * shape[1], shape[0] * shape[1]))

    row = []
    col = []
    val = []
    for iy in range(shape[0]):
        for ix in range(shape[1]):
            if iy < alp or iy > shape[0] - 1 - alp:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy, ix, shape))
                val.append(1.0)
    Pa_mat = sparse.csr_matrix((val, (row, col)), shape=(shape[0] * shape[1], shape[0] * shape[1]))

    a = data_gau.copy()
    a = a[::-1, :]
    a = a.reshape((shape[0] * shape[1], 1))
    a = sparse.csr_matrix(a)
    a = linalg.spsolve(L_mat.transpose() * L_mat + lmb_b * Pa_mat, lmb_b * Pa_mat * a)

    row = []
    col = []
    val = []
    for iy in range(shape[0]):
        for ix in range(shape[1]):
            if ix < alp or ix > shape[1] - 1 - alp:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy, ix, shape))
                val.append(1.0)
    Pb_mat = sparse.csr_matrix((val, (row, col)), shape=(shape[0] * shape[1], shape[0] * shape[1]))

    b = data_gau.copy()
    b = b[:, ::-1]
    b = b.reshape((shape[0] * shape[1], 1))
    b = sparse.csr_matrix(b)
    b = linalg.spsolve(L_mat.transpose() * L_mat + lmb_b * Pb_mat, lmb_b * Pb_mat * b)

    row = []
    col = []
    val = []
    for iy in range(shape[0]):
        for ix in range(shape[1]):
            if iy < alp or iy > shape[0] - 1 - alp or ix < alp or ix > shape[1] - 1 - alp:
                row.append(get_idx(iy, ix, shape))
                col.append(get_idx(iy, ix, shape))
                val.append(1.0)
    Pc_mat = sparse.csr_matrix((val, (row, col)), shape=(shape[0] * shape[1], shape[0] * shape[1]))

    c = (Pb_mat.diagonal() * a).reshape(shape[0], shape[1])[:, ::-1] + (Pa_mat.diagonal() * b).reshape(shape[0], shape[1])[::-1, :]
    c[0: alp, 0: alp] /= 2.0
    c[shape[0] - alp: shape[0], 0: alp] /= 2.0
    c[0: alp, shape[1] - alp: shape[1]] /= 2.0
    c[shape[0] - alp: shape[0], shape[1] - alp: shape[1]] /= 2.0

    c = c.reshape((shape[0] * shape[1], 1))
    c = sparse.csr_matrix(c)
    c = linalg.spsolve(L_mat.transpose() * L_mat + lmb_b * Pc_mat, lmb_b * Pc_mat * c)

    a = a.reshape(shape[0], shape[1]) * scale
    b = b.reshape(shape[0], shape[1]) * scale
    c = c.reshape(shape[0], shape[1]) * scale

    row1 = np.append(np.append(c, a, axis=1), c, axis=1)
    row2 = np.append(np.append(b, data, axis=1), b, axis=1)
    out = np.append(np.append(row1, row2, axis=0), row1, axis=0)
    out = out[shape[0] // 2: 2 * shape[0] + shape[0] // 2, shape[1] // 2: 2 * shape[1] + shape[1] // 2]
    return out


def f_filter(nt, dt, f_cut, order, decay, is_lowpass=True, max_clip=400.0):
    f = np.fft.fftfreq(nt, dt)
    f = np.abs(f)
    x = f - f_cut if is_lowpass else f_cut - f
    x = np.clip(x, -max_clip, max_clip)
    m = - (decay / order) * np.log2(1 + 2 ** (order * x))
    m = 2 ** m
    return m


def fk_filter(nt, nx, dt, dx, v_cut, order, decay, is_lowpass=True, max_clip=400.0):
    f = np.fft.fftfreq(nt, dt)[:, None].repeat(nx, axis=1)
    k = np.fft.fftfreq(nx, dx)[None, :].repeat(nt, axis=0)
    v = f / (k + 1e-10)
    v = np.abs(v)

    x = v - v_cut if is_lowpass else v_cut - v
    x = np.clip(x, -max_clip, max_clip)

    m = - (decay / order) * np.log2(1 + 2 ** (order * x))
    m = 2 ** m
    # freq_mask = f_filter(nt, dt, 5, 2, 4, is_lowpass=False)[:, None].repeat(nx, axis=1)
    # m = freq_mask * m
    return m


def f_filtering(data, mask, is_zeroout=False):
    shape = data.shape
    nt = shape[0] if len(shape) == 1 else shape[len(shape) - 2]
    m_f = mask.copy()
    if is_zeroout:
        m_f[0] = 0.0

    if len(shape) == 1:
        data_unfold = data.copy()[None, :]
    else:
        data_unfold = np.swapaxes(data, len(shape) - 2, len(shape) - 1)
        data_unfold = data_unfold.reshape((-1, nt))

    if len(shape) > 1:
        shape = list(shape)
        tmp = shape[-1]
        shape[-1] = shape[-2]
        shape[-2] = tmp
        shape = tuple(shape)

    data_unfold = np.fft.fft(data_unfold, axis=-1)
    data_unfold *= m_f[None]
    data_unfold = np.fft.ifft(data_unfold, axis=-1)
    data_unfold = np.real(data_unfold)

    if len(shape) > 1:
        data_unfold = data_unfold.reshape(shape)
    else:
        data_unfold = data_unfold.squeeze()
    if len(shape) > 1:
        data_unfold = np.swapaxes(data_unfold, len(shape) - 2, len(shape) - 1)
    return data_unfold
