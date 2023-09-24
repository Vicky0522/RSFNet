import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from tqdm import tqdm

from basicsr.archs.inception import InceptionV3
from basicsr.utils.registry import METRIC_REGISTRY


def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
    # we may not resize the input, but in [rosinality/stylegan2-pytorch] it
    # does resize the input.
    inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
    inception = nn.DataParallel(inception).eval().to(device)
    return inception


def calculate_fid_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for
            generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an
               representative data set.
        sigma2 (np.array): The covariance matrix over activations,
            precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid


@METRIC_REGISTRY.register()
class CalculateFid(object):
    def __init__(self device="cuda"):
        inception_model_fid = load_patched_inception_v3()
        inception_model_fid.to(device)
        inception_model_fid.eval()

        self.device = device
        self.inception_model_fid = inception_model_fid

        self.fake_acts_set = []
        self.real_acts_set = []

    @torch.no_grad()
    def extract_inception_features(img, len_generator=None):
        """Extract inception features.

        Args:
            data_generator (generator): A data generator.
            inception (nn.Module): Inception model.
            len_generator (int): Length of the data_generator to show the
                progressbar. Default: None.
            device (str): Device. Default: cuda.

        Returns:
            Tensor: Extracted features.
        """
        data = pyutils.img2tensor(img, 255., device=self.device)
        feature = inception(data)[0].view(data.shape[0], -1)
        return feature.cpu().numpy()

    def add_data(self, img, img2)
        fake_act = self.extract_inception_features(img)
        real_act = self.extract_inception_features(img2)
        self.fake_acts_set.append(fake_act)
        self.real_acts_set.append(real_act)

    def calculate_activation_statistics(self, acts_set):
        """Calculation of the statistics used by the FID.
        Params:
        -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        mu = np.mean(acts_set, axis=0)
        sigma = np.cov(acts_set, rowvar=False)
        return mu, sigma

    def calculate(self):
        fake_acts_set = np.concatenate(self.fake_acts_set, 0)
        real_acts_set = np.concatenate(self.real_acts_set, 0)

        real_mu, real_sigma = self.calculate_activation_statistics(real_acts_set)
        fake_mu, fake_sigma = self.calculate_activation_statistics(fake_acts_set)
        fid_score = calculate_fid_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        self.fake_acts_set = []
        self.real_acts_set = []
        return fid_score
