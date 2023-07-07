import torch
from tqdm import tqdm

# First Impressions V2 train set mean and std values
fi_eGeMAPS_mean = torch.tensor([
    3.0264e+01, 2.2181e-01, 2.5627e+01, 3.0327e+01, 3.4389e+01, 8.7614e+00,
    2.7738e+02, 3.7881e+02, 1.3259e+02, 1.8899e+02, 1.2498e+00, 6.4475e-01,
    5.5138e-01, 1.0799e+00, 1.8971e+00, 1.3457e+00, 1.7504e+01, 1.0418e+01,
    1.1953e+01, 7.0613e+00, 9.0333e-01, 8.7972e-01, 2.0565e+01, 8.3367e-01,
    -1.5729e+00, 5.9070e-01, 7.7730e+00, 6.0822e-01, -6.2394e+00, -1.4025e+00,
    5.9561e-02, 1.6119e+00, 1.4631e+00, 6.9436e-01, 3.0220e+00, 6.3954e-01,
    4.0100e+00, 1.0370e+00, 1.5863e+01, 5.6314e-01, 5.7538e+02, 4.2570e-01,
    1.1935e+03, 2.1568e-01, -5.6035e+01, -1.6077e+00, 1.5921e+03, 1.7843e-01,
    9.1540e+02, 3.7798e-01, -5.5353e+01, -1.3962e+00, 2.6026e+03, 1.1820e-01,
    8.5064e+02, 4.4309e-01, -5.8284e+01, -1.2654e+00, -9.8316e+00, -8.7251e-01,
    1.9423e+01, 4.7845e-01, 4.3844e-02, 4.9679e-01, -1.7817e-02, -6.5778e-01,
    1.0400e+00, 7.5216e-01, 2.4323e+01, 4.9244e-01, -3.0196e+00, 9.2110e+00,
    8.4160e+00, 6.1843e+00, -8.4154e+00, 2.3800e-01, -3.9070e+00, 1.1524e+01,
    1.5815e-02, -8.0057e-03, 4.4074e-01, 3.4899e+00, 2.1854e+00, 4.1770e-01,
    3.8554e-01, 1.1914e-01, 1.2242e-01, -2.4863e+01
])
fi_eGeMAPS_std = torch.tensor([
    5.2181e+00, 7.6056e-02, 6.5450e+00, 5.6324e+00, 5.6652e+00, 5.2366e+00,
    1.2812e+02, 2.3054e+02, 6.6159e+01, 1.6824e+02, 6.1585e-01, 1.5276e-01,
    3.6629e-01, 5.8105e-01, 9.3020e-01, 6.7781e-01, 8.7167e+00, 5.3331e+00,
    5.9689e+00, 3.4606e+00, 6.5497e-01, 2.2223e-01, 6.2634e+00, 1.0276e+01,
    8.4901e+00, 1.2328e+02, 7.5030e+00, 1.0788e+02, 8.7715e+00, 7.4784e+01,
    2.0744e-02, 2.5745e-01, 2.2530e-01, 9.7275e-02, 2.2082e+00, 6.3851e+01,
    3.5933e+00, 1.0824e+02, 5.4092e+00, 1.2046e+01, 9.4174e+01, 6.3964e-02,
    1.0129e+02, 4.6412e-02, 2.5680e+01, 6.2079e-01, 1.0814e+02, 2.3429e-02,
    1.0560e+02, 7.2894e-02, 2.2825e+01, 3.9420e-01, 1.1934e+02, 1.4683e-02,
    1.0357e+02, 8.9094e-02, 2.2261e+01, 2.9110e-01, 4.4124e+00, 1.5401e+01,
    4.7334e+00, 1.7188e-01, 3.9308e-02, 3.2508e+01, 7.0149e-03, 1.9835e+01,
    7.1772e-01, 1.6505e-01, 6.2107e+00, 9.5421e-01, 9.2457e+00, 1.0043e+03,
    8.1555e+00, 5.1247e+02, 9.7285e+00, 9.5642e+01, 4.8929e+00, 5.6538e+00,
    4.2924e-02, 6.5074e-03, 4.9300e-01, 7.4664e-01, 6.9000e-01, 4.7664e-01,
    3.0668e-01, 3.4708e-01, 1.2347e-01, 6.4658e+00
])
fi_pyaa_mt_mean = torch.tensor([
    7.0353e-02, 1.3350e-02, 3.0208e+00, 1.3539e-01, 1.6000e-01, 5.0237e-01,
    1.0655e-02, 1.0587e-01, -2.7089e+01, 1.8061e+00, -1.6297e-01, -8.4079e-02,
    -2.5610e-01, 4.5949e-02, -2.4631e-02, -2.4509e-02, -7.0880e-02,
    -6.5149e-02, -7.6019e-02, -2.6637e-02, -7.9259e-02, 1.7896e-02, 7.2872e-03,
    2.9653e-02, 1.5710e-02, 2.1303e-02, 1.2832e-02, 2.8439e-02, 4.3710e-03,
    8.3135e-03, 1.3089e-02, 1.9858e-02, 7.8439e-03, 2.3700e-02, -5.1265e-06,
    -2.2364e-06, -1.0083e-04, -1.3634e-05, -1.7085e-05, -6.4243e-05,
    4.0614e-05, -8.4025e-06, -1.5790e-03, 1.6052e-04, -2.0536e-05, 3.0857e-05,
    9.2350e-05, -1.4916e-04, 9.3591e-06, 5.5354e-05, -1.3509e-05, 9.5243e-05,
    -1.0121e-05, -2.9499e-05, 3.4029e-05, -2.8230e-06, -4.3167e-07,
    -3.8042e-06, 7.8857e-06, 7.3577e-06, -5.2958e-06, 5.0350e-06, -1.5153e-06,
    1.8563e-06, -8.1202e-07, 1.5718e-08, 4.3633e-07, -8.0098e-07, 5.2879e-02,
    1.4253e-02, 2.8948e-01, 5.3103e-02, 3.1509e-02, 4.5177e-01, 7.7460e-03,
    9.3085e-02, 2.3362e+00, 8.2419e-01, 7.5047e-01, 5.0541e-01, 4.2749e-01,
    3.6587e-01, 3.2583e-01, 2.9587e-01, 2.8360e-01, 2.6838e-01, 2.6654e-01,
    2.5850e-01, 2.5349e-01, 2.1233e-02, 1.0285e-02, 2.8980e-02, 1.9644e-02,
    2.2091e-02, 1.7000e-02, 2.7865e-02, 7.0962e-03, 1.1106e-02, 1.4633e-02,
    2.0552e-02, 1.0805e-02, 1.3007e-02, 5.1988e-02, 1.4370e-02, 4.0098e-01,
    5.3203e-02, 3.7615e-02, 4.6402e-01, 9.6000e-03, 9.9581e-02, 1.9124e+00,
    8.0183e-01, 6.3161e-01, 4.8571e-01, 4.3396e-01, 3.7470e-01, 3.4563e-01,
    3.1932e-01, 3.0610e-01, 2.9262e-01, 2.8721e-01, 2.7853e-01, 2.7149e-01,
    2.4554e-02, 1.2609e-02, 3.2901e-02, 2.3104e-02, 2.5574e-02, 2.0404e-02,
    3.1666e-02, 8.6976e-03, 1.3511e-02, 1.7398e-02, 2.3705e-02, 1.3168e-02,
    1.4405e-02
])
fi_pyaa_mt_std = torch.tensor([
    3.5798e-02, 1.4878e-02, 1.8061e-01, 3.5469e-02, 1.9975e-02, 2.9329e-01,
    7.5557e-03, 5.9725e-02, 2.8884e+00, 7.5670e-01, 5.6362e-01, 3.9366e-01,
    3.1442e-01, 2.5717e-01, 2.3000e-01, 2.0826e-01, 1.8874e-01, 1.9586e-01,
    1.8301e-01, 1.8611e-01, 1.8045e-01, 1.5404e-02, 6.6397e-03, 2.5301e-02,
    1.3341e-02, 1.8193e-02, 1.0778e-02, 3.0062e-02, 4.7331e-03, 7.1520e-03,
    1.0244e-02, 1.6379e-02, 7.0856e-03, 9.6615e-03, 6.1375e-03, 2.0425e-03,
    3.0371e-02, 5.7735e-03, 3.1447e-03, 4.8837e-02, 2.1934e-03, 1.0455e-02,
    3.1521e-01, 8.7515e-02, 8.0060e-02, 5.3705e-02, 4.4766e-02, 3.7867e-02,
    3.3746e-02, 3.0362e-02, 2.8909e-02, 2.7886e-02, 2.8135e-02, 2.7088e-02,
    2.6715e-02, 2.6708e-03, 1.3934e-03, 3.4995e-03, 2.5181e-03, 2.7074e-03,
    2.1629e-03, 3.7259e-03, 1.0751e-03, 1.4552e-03, 1.8349e-03, 2.4968e-03,
    1.4394e-03, 1.3695e-03, 3.4161e-02, 1.4550e-02, 1.4073e-01, 2.5614e-02,
    8.5535e-03, 2.1519e-01, 1.3116e-02, 5.4510e-02, 1.4191e+00, 2.9056e-01,
    2.5530e-01, 1.6711e-01, 1.3478e-01, 1.0785e-01, 9.2961e-02, 8.2842e-02,
    8.2231e-02, 8.2087e-02, 8.8047e-02, 8.8501e-02, 8.8492e-02, 1.7365e-02,
    9.9829e-03, 2.0740e-02, 1.6689e-02, 1.6407e-02, 1.5162e-02, 2.4005e-02,
    8.5068e-03, 1.0541e-02, 1.1631e-02, 1.4863e-02, 1.0375e-02, 5.0314e-03,
    3.2098e-02, 1.4390e-02, 1.9958e-01, 2.3893e-02, 1.1191e-02, 2.2945e-01,
    1.7373e-02, 5.8545e-02, 1.0603e+00, 2.6953e-01, 1.9277e-01, 1.3999e-01,
    1.2343e-01, 9.7122e-02, 8.7637e-02, 7.7984e-02, 7.4859e-02, 7.2507e-02,
    7.3502e-02, 7.2810e-02, 7.2363e-02, 1.8217e-02, 1.1447e-02, 2.1504e-02,
    1.7899e-02, 1.7311e-02, 1.6752e-02, 2.5429e-02, 9.9380e-03, 1.2012e-02,
    1.2804e-02, 1.5614e-02, 1.1814e-02, 5.1894e-03
])
fi_utt_eGeMAPS_lld_mean = torch.tensor([
    1.3172e+00, -8.5697e+00, 1.7732e+01, 3.7983e-02, -1.5957e-02, 9.5845e-01,
    2.1413e+01, -1.8157e+00, 8.3333e+00, -6.7366e+00, 2.4144e+01, 4.1621e-02,
    1.1201e+00, 2.5096e+00, 3.2088e+00, 1.2767e+01, 5.7517e+02, 1.1675e+03,
    -4.2725e+01, 1.5890e+03, -5.0210e+01, 2.5960e+03, -5.3349e+01
])
fi_utt_eGeMAPS_lld_std = torch.tensor([
    1.0574e+00, 8.9053e+00, 1.0423e+01, 5.0413e-02, 1.5717e-02, 1.1126e+00,
    1.4715e+01, 1.7309e+01, 1.7812e+01, 1.7976e+01, 1.4195e+01, 8.8883e-02,
    1.0937e+00, 4.9061e+00, 1.3710e+01, 1.4358e+01, 2.6162e+02, 2.8341e+02,
    7.6037e+01, 3.0037e+02, 7.1796e+01, 3.2272e+02, 7.0055e+01
])
fi_eGeMAPSv02_mean = torch.tensor([
    1.2502813e+00, -8.5120935e+00, 1.7671463e+01, 3.6979329e-02,
    -1.5389772e-02, 9.0379250e-01, 2.0567286e+01, -1.5753851e+00,
    7.7754025e+00, -6.2401347e+00, 2.3333593e+01, 3.1896140e-02, 9.5673466e-01,
    2.3294139e+00, 3.1215832e+00, 1.2369150e+01, 5.9987286e+02, 1.1645104e+03,
    -4.8352169e+01, 1.6170449e+03, 9.0268024e+02, -5.5322937e+01,
    2.6324375e+03, 8.4030292e+02, -5.8399246e+01
])
fi_eGeMAPSv02_std = torch.tensor([
    1.0490705e+00, 9.2247686e+00, 1.0745084e+01, 5.3866968e-02, 1.5901020e-02,
    1.0999205e+00, 1.5021754e+01, 1.7405146e+01, 1.7598169e+01, 1.7972860e+01,
    1.4750662e+01, 8.0683932e-02, 9.8545361e-01, 4.8984027e+00, 1.3756578e+01,
    1.4520527e+01, 2.6754742e+02, 2.9092075e+02, 7.9702469e+01, 3.0441605e+02,
    3.5762766e+02, 7.5428749e+01, 3.2853674e+02, 3.8451575e+02, 7.3620964e+01
])
fi_au_mean = torch.tensor([
    0.33668621, 0.13915953, 0.38113375, 0.08974244, 0.29177619, 0.57494336,
    0.10040912, 0.47771274, 0.48285216, 0.45467833, 0.19797922, 0.57138002,
    0.14053742, 0.20013267, 0.67406506, 0.57241297, 0.3697266, 0.26421491,
    0.26582336, 0.19971083, 0.48215325, 0.160271, 0.30328639, 0.09744758,
    0.31978551, 0.22113819, 0.3687779, 0.18917823, 0.31193846, 0.13581071,
    0.13286627, 0.33433577, 0.22353263, 0.01257307, 0.29282676
])
fi_au_std = torch.tensor([
    0.60922607, 0.35848543, 0.59989241, 0.26159211, 0.47718787, 0.73845894,
    0.23984702, 0.63615664, 0.61309048, 0.62698945, 0.3994489, 0.58512136,
    0.28784035, 0.41074128, 0.68013687, 0.62144567, 0.68748725, 0.44091427,
    0.44177065, 0.39978296, 0.49968139, 0.3668572, 0.45967788, 0.29656627,
    0.46639333, 0.41501336, 0.48247358, 0.39165013, 0.46328485, 0.34258745,
    0.33943015, 0.47175774, 0.41661228, 0.11142255, 0.45505961
])


def get_mean_std(loader, ndim):

    # VAR[X] = E[X**2] - E[X]**2
    if ndim == 2:
        # vectors - (B, C)
        dim = [0]
    elif ndim == 3:
        # time series - (B, T, C)
        dim = [0, 1]
    elif ndim == 4:
        # images - (B, C, H, W)
        dim = [0, 2, 3]
    elif ndim == 5:
        # videos - (B, T, C, H, W)
        dim = [0, 1, 3, 4]
    else:
        raise NotImplementedError()

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader, total=len(loader)):
        channels_sum += torch.mean(data, dim=dim)
        channels_squared_sum += torch.mean(data**2, dim=dim)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    return mean, std


def norm(x, mean, std):
    # (B, F) and (B, T, F)
    return (torch.FloatTensor(x) - mean) / (std + torch.finfo(torch.float32).eps)