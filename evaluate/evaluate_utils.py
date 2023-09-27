import torch
import torch.nn.functional as F
import numpy as np

def rmse(predictions, targets):
    # Compute the differences for the last dimension
    differences = predictions[:, 0, :] - targets[:, 0, :]

    # Square the differences and compute the mean along the last dimension, then take the square root
    rmse_values = np.sqrt((differences ** 2).mean(axis=-1))

    return rmse_values

def psnr(img1, img2, max_val=1.0):
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()


    mse = F.mse_loss(img1, img2, reduction='none').mean([1, 2])

    psnr_val = 10 * torch.log10(max_val ** 2 / mse)

    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        return psnr_val.numpy()

    return psnr_val

def cosine_sim(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def fre_cosine(signal1, signal2):
    # check numpy or tensor
    if isinstance(signal1, np.ndarray) and isinstance(signal2, np.ndarray):
        spectrum1 = np.fft.fft(signal1, axis=-1)
        spectrum2 = np.fft.fft(signal2, axis=-1)
        cos_sim_func = cosine_sim
    elif isinstance(signal1, torch.Tensor) and isinstance(signal2, torch.Tensor):
        signal1 = signal1.cpu()
        signal2 = signal2.cpu()

        spectrum1 = torch.fft.fft(signal1, dim=-1)
        spectrum2 = torch.fft.fft(signal2, dim=-1)
        cos_sim_func = torch.nn.functional.cosine_similarity
    else:
        raise ValueError("Both inputs should be either numpy arrays or torch tensors")

    # calculate similarity
    B = signal1.shape[0]
    similarities = np.zeros(B) if isinstance(signal1, np.ndarray) else torch.zeros(B)
    for i in range(B):
        if isinstance(signal1, np.ndarray):
            similarities[i] = cos_sim_func(np.abs(spectrum1[i, 0, :]), np.abs(spectrum2[i, 0, :]))
        else:
            similarities[i] = cos_sim_func(torch.abs(spectrum1[i, :]), torch.abs(spectrum2[i, :]))

    return similarities


def eval_all(signal1, signal2):
    rmse_all=rmse(signal1, signal2)
    psnr_all=psnr(signal1, signal2)
    fre_cos_all=fre_cosine(signal1, signal2)
    return [rmse_all,psnr_all,fre_cos_all]
