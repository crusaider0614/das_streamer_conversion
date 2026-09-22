"""Frechet Inception Distance, for G(A) against real B.

Ported from the fwi_mk2/velocity-gan implementation so the numbers stay
comparable with the earlier work.  Three things differ, and they all come
from the data rather than from taste:

  * the input here is already in [-1, 1] (the dataset divides by
    `value_range` and the generator ends in tanh), so IN_RANGE says so and
    the [0, 1] -> [-1, 1] rescale Inception wants is done once, here,
    instead of twice.

  * the patches are single channel and 128 x 128, so they are replicated to
    three channels and resized to 299 the same way the original did.  A
    seismic patch is not a photograph and Inception has never seen one; the
    activations are still a fixed, smooth, high-dimensional summary, which
    is all the Frechet distance needs, but the absolute value means nothing
    on its own.  Read it against the das-vs-str baseline that
    inference/eval_cut_checkpoints.py prints, never against a published
    image-model FID.

  * 2048 x 2048 covariances are estimated from at most a few thousand
    patches, so they are rank deficient and the distance is biased upward.
    The bias is roughly constant at fixed sample count, which is why both
    sets are capped at the same MAX_SAMPLES - comparing two FIDs computed
    from different numbers of patches is meaningless.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


class InceptionNetwork(nn.Module):
    """InceptionV3 truncated at Mixed_7c, global-pooled to 2048."""

    mixed_7c_output: torch.Tensor

    def __init__(self):
        super(InceptionNetwork, self).__init__()
        from torchvision.models import inception_v3
        try:
            from torchvision.models import Inception_V3_Weights
            self.network = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1)
        except ImportError:                     # torchvision < 0.13
            self.network = inception_v3(pretrained=True)
        self.network.Mixed_7c.register_forward_hook(self.output_hook)

    def output_hook(self, module, input, output):
        self.mixed_7c_output = output

    def forward(self, x):
        assert x.shape[1:] == (3, 299, 299), \
            "expected (N, 3, 299, 299), got {}".format(tuple(x.shape))
        self.network(x)
        activation = F.adaptive_avg_pool2d(self.mixed_7c_output, (1, 1))
        return activation.view(x.shape[0], 2048)


def get_activation(data, batch_size, device, network=None, in_range=(-1.0, 1.0)):
    """(N, 1, H, W) in `in_range` -> (N, 2048) float32.

    `network` lets a caller build the Inception once and score many
    checkpoints with it; left None it is built and thrown away per call, as
    in the original.
    """
    data = torch.as_tensor(np.asarray(data, dtype=np.float32))
    if data.ndim == 3:
        data = data[:, None]
    n_data = data.shape[0]
    n_batch = int(np.ceil(n_data / batch_size))

    owned = network is None
    if owned:
        network = InceptionNetwork().eval().to(device)

    lo, hi = in_range
    scale = 2.0 / (hi - lo)                     # -> [-1, 1], what Inception wants

    activation = np.zeros((n_data, 2048), dtype=np.float32)
    for batch_idx in range(n_batch):
        s = batch_size * batch_idx
        e = min(batch_size * (batch_idx + 1), n_data)

        batch = data[s:e].to(device)
        batch = scale * (batch - lo) - 1.0
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear",
                              align_corners=False)
        batch = batch.expand(-1, 3, -1, -1)
        with torch.no_grad():
            activation[s:e, :] = network(batch).cpu().numpy()
        if device != "cpu":
            torch.cuda.empty_cache()

    if owned:
        del network
        if device != "cpu":
            torch.cuda.empty_cache()
    return activation


def calculate_activation_statistics(data, batch_size, device, network=None,
                                    in_range=(-1.0, 1.0)):
    activation = get_activation(data, batch_size, device, network, in_range)
    mu = np.mean(activation, axis=0)
    sigma = np.cov(activation, rowvar=False)
    del activation
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("imaginary component {}".format(
                np.max(np.abs(covmean.imag))))
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2 * np.trace(covmean))


def calculate_fid(data1, data2, batch_size, device, network=None,
                  in_range=(-1.0, 1.0)):
    mu1, s1 = calculate_activation_statistics(data1, batch_size, device,
                                              network, in_range)
    mu2, s2 = calculate_activation_statistics(data2, batch_size, device,
                                              network, in_range)
    return calculate_frechet_distance(mu1, s1, mu2, s2)
