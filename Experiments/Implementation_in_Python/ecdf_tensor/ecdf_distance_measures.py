from ecdf_tensor.configuration import ECDFDistanceMeasureConfiguration as Conf

from enum import Enum
from typing import Tuple
from tqdm import tqdm

import random
import torch
import numpy as np


class ECDFDistanceMeasures(Enum):
    WASSERSTEIN_DISTANCE = 1
    CRAMER_VON_MISES_DISTANCE = 2
    KUIPER_DISTANCE = 3
    ANDERSON_DARLING_DISTANCE = 4
    KOLMOGOROV_SMIRNOV_DISTANCE = 5
    DTS_DISTANCE = 6

class ECDFDistanceMeasure:
    DISTANCE_TYPE = None

    def measure_metric_p_value(self, data_1: np.ndarray, data_2: np.ndarray, filter_p_value: bool = True,
                               bootstrap_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """The function returns measures on a set of input arrays. The
        arrays are assumed in the shape of (number, width, height,
        channels).

        :param data_1: the first dataset to be compared in the form of numpy ndarray
        :param data_2: the second dataset to be compared in the form numpy ndarray
        :param filter_p_value: Flag to Filter distance based on pVal
            less than 0.05. Default value is True.
        :param bootstrap_samples: is the optional input for pVal
            accuracy. Default value is 1000.
        :return: The outputs are Distance metric and P value.
        """
        if len(data_1.shape) == 4:
            [nA, w, h, c] = data_1.shape
            nB = data_2.shape[0]
            D = np.zeros(w * h * c)
            pVal = np.zeros(w * h * c)
        elif len(data_1.shape) == 3:
            [nA, w, h] = data_1.shape
            c = 1
            nB = data_2.shape[0]
            D = np.zeros(w * h)
            pVal = np.zeros(w * h)
        # TODO: the == 2 case seems to be required for 1d multi feature
        #  data -> required below in the gpu-version method as well
        #  Not certain whether this is the intended way to handle such
        #  input data.
        elif len(data_1.shape) == 2:
            nA, w = data_1.shape
            h = c = 1
            nB = data_2.shape[0]
            D = np.zeros(w * h)
            pVal = np.zeros(w * h)
        else:
            nA = data_1.shape[0]
            w = 1
            h = 1
            c = 1
            nB = data_2.shape[0]
            D = np.zeros(w*h)
            pVal = np.zeros(w*h)

        xxx_2 = np.array([yy.flatten() for yy in data_1])
        yyy_2 = np.array([yy.flatten() for yy in data_2])

        if w == h == c == 1:
            pVal, D = self._compute_distance_p_value(xxx_2.flatten(), yyy_2.flatten(), bootstrap_samples)
        else:
            for kk in tqdm(range(1, w * h * c)):
                pVal[kk], D[kk] = self._compute_distance_p_value(xxx_2[:nA, kk], yyy_2[:nB, kk], bootstrap_samples)

            if filter_p_value:
                D[pVal > Conf.p_value_alpha_threshold] = 0
                pVal[pVal > Conf.p_value_alpha_threshold] = 0

        return pVal, D

    def measure_metric_p_value_gpu(self, data_1: np.ndarray, data_2: np.ndarray, filter_p_value: bool = True,
                                   bootstrap_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """The function returns measures on a set of input arrays. The
        arrays are assumed in the shape of (number, width, height,
        channels). The GPU is used for the distance computations.

        :param data_1: the first dataset to be compared in the form of numpy ndarray
        :param data_2: the second dataset to be compared in the form numpy ndarray
        :param filter_p_value: Flag to Filter distance based on pVal
            less than 0.05. Default value is True.
        :param bootstrap_samples: is the optional input for pVal
            accuracy. Default value is 1000.
        :return: Tuple of the p-Value (is a measure of suitability of
            the metric) and the Tensor including the distances.
        """
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
            print("WARNING: cant find cuda. Computation will not be accelerated.")
        
        if len(data_1.shape) == 4:
            [nA, w, h, c] = data_1.shape
            nB = data_2.shape[0]
            D = torch.zeros(w * h * c)
            pVal = torch.zeros(w * h * c)
        elif len(data_1.shape) == 3:
            [nA, w, h] = data_1.shape
            c = 1
            nB = data_2.shape[0]
            D = torch.zeros(w * h)
            pVal = torch.zeros(w * h)
        else:
            nA = data_1.shape[0]
            w = 1
            h = 1
            c = 1
            nB = data_2.shape[0]
            D = torch.zeros(w * h)
            pVal = torch.zeros(w * h)

        xxx_2 = torch.tensor(np.array([yy.flatten() for yy in data_1])).to(DEVICE)
        yyy_2 = torch.tensor(np.array([yy.flatten() for yy in data_2])).to(DEVICE)

        if w == h == c == 1:
            pVal, D = self._compute_distance_p_value_gpu(xxx_2.flatten(), yyy_2.flatten(), w, h, c, DEVICE, bootstrap_samples)
        else:
            pVal, D = self._compute_distance_p_value_gpu(xxx_2, yyy_2, w, h, c, DEVICE, bootstrap_samples)

            if filter_p_value:
                compare = pVal > Conf.p_value_alpha_threshold
                if not (w == h == c == 1):
                    D[compare.nonzero()] = 0
                    pVal[compare.nonzero()] = 0

        return pVal, D

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        """This function returns the distance between the two given
        numpy arrays.

        :param data_1: flattened numpy ndarray 
        :param data_2: flattened numpy ndarray 
        :return: distance measures.
        """
        raise NotImplementedError

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        """This function returns the distance between given numpy arrays
        and corresponding pVal.

        :param data_1: the first dataset to be compared in the form of numpy ndarray
        :param data_2: the second dataset to be compared in the form numpy ndarray
        :param bootstrap_samples: is the optional input for pVal
            accuracy. Default value is 1000.
        :return: Tuple of the p-Value as a measure of suitability of the
            metric and the distance measure.
        """
        raise NotImplementedError

    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                              color: int, device: str) -> torch.Tensor:
        """This function returns the distance between the two given
        PyTorch Tensors.

        :param data_1: the first dataset to be compared in the form of torch tensor
        :param data_2: the second dataset to be compared in the form torch tensor
        :param width: width of datapoint (in case of image shaped data)
        :param height: height of datapoint (in case of image shaped data)
        :param color: color of datapoint (in case of image shaped data)
        :param device: device of torch tensor ('cpu' or 'cuda')
        :return: Tensor of the distances.
        """
        raise NotImplementedError

    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function returns the distance between the two given
        PyTorch Tensors and the corresponding p-Values.

        :param data_1: the first dataset to be compared in the form of torch tensor
        :param data_2: the second dataset to be compared in the form torch tensor
        :param width: width of datapoint (in case of image shaped data)
        :param height: height of datapoint (in case of image shaped data)
        :param color: color of datapoint (in case of image shaped data)
        :param device: device of torch tensor ('cpu' or 'cuda')
        :param bootstrap_samples: is the optional input for pVal
            accuracy. Default value is 1000.
        :return: Tuple of the p-Value (is a measure of suitability of
            the metric) and the Tensor including the distances.
        """
        raise NotImplementedError


class WassersteinDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.WASSERSTEIN_DISTANCE

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        # print(XX)
        XY = np.concatenate([data_1, data_2])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res = 0
        E_CDF = 0
        F_CDF = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            height = abs(F_CDF - E_CDF)
            width = XY_Sorted[ii + 1] - XY_Sorted[ii]
            Res = Res + (height ** power) * width

        return Res

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        WD = self._compute_distance(data_1, data_2)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = np.concatenate([data_1, data_2])
        reps = 0
        bigger = 0
        for ii in range(1, bootstrap_samples):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            boost_WD = self._compute_distance(comb[e], comb[f])
            if (boost_WD > WD):
                bigger = 1 + bigger

        pVal = bigger / bootstrap_samples

        return pVal, WD

    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                              color: int, device: str) -> torch.Tensor:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = torch.cat([data_1, data_2]).to(device)
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        if width == height == color == 1:
            ar_x = torch.tensor(X2).to(device)
            ar_y = torch.tensor(Y2).to(device)
            S_Ind_torch = torch.argsort(XY).to(device)
            XY_sorted = XY[S_Ind_torch]
            ar_x_sorted = ar_x[S_Ind_torch]
            ar_y_sorted = ar_y[S_Ind_torch]
        else:
            ar_x = np.tile([X2], (width * height * color, 1))
            ar_x = torch.tensor(ar_x).to(device)
            ar_x = torch.transpose(ar_x, 1, 0)

            ar_y = np.tile([Y2], (width * height * color, 1))
            ar_y = torch.tensor(ar_y).to(device)
            ar_y = torch.transpose(ar_y, 1, 0)

            S_Ind_torch = torch.argsort(XY, dim=0).to(device)
            XY_sorted = torch.gather(XY, 0, S_Ind_torch)
            ar_x_sorted = torch.gather(ar_x, 0, S_Ind_torch)
            ar_y_sorted = torch.gather(ar_y, 0, S_Ind_torch)

        power = 1

        E_CDF = torch.cumsum(ar_x_sorted[:-1], dim=0)
        F_CDF = torch.cumsum(ar_y_sorted[:-1], dim=0)
        height_wd = abs(F_CDF - E_CDF)
        width_wd = XY_sorted[1:] - XY_sorted[0:-1]
        Res = (height_wd ** power) * width_wd
        Res = torch.sum(Res, dim=0)

        return Res

    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        WD = self._compute_distance_gpu(data_1, data_2, width, height, color, device)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = torch.cat((data_1, data_2)).to(device)
        reps = 0
        bigger = torch.zeros([width * height * color])
        for ii in tqdm(range(1, bootstrap_samples)):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)

            if len(comb.shape) > 1:
                xa = comb[e, :]
                xb = comb[f, :]
            else:
                xa = comb[e]
                xb = comb[f]

            boost_WD = self._compute_distance_gpu(xa, xb, width, height, color, device)
            compare = boost_WD > WD
            bigger[compare.nonzero()] += 1

        pVal = bigger / bootstrap_samples

        return pVal, WD


class CramerVonMisesDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.CRAMER_VON_MISES_DISTANCE

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = np.concatenate([data_1, data_2])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res = 0
        height = 0
        E_CDF = 0
        F_CDF = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            height = abs(F_CDF - E_CDF)
            if XY_Sorted[ii + 1] != XY_Sorted[ii]:
                Res = Res + height ** power

        CVM_Dist = Res

        return CVM_Dist

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        CVM = self._compute_distance(data_1, data_2)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = np.concatenate([data_1, data_2])
        reps = 0
        bigger = 0
        for ii in range(1, bootstrap_samples):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            boost_CVM = self._compute_distance(comb[e], comb[f])
            if (boost_CVM > CVM):
                bigger = 1 + bigger

        pVal = bigger / bootstrap_samples

        return pVal, CVM
    
    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height_data: int,
                              color: int, device: str) -> torch.Tensor:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = torch.cat([data_1, data_2]).to(device)
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        if width == height_data == color == 1:
            ar_x = torch.tensor(X2).to(device)
            ar_y = torch.tensor(Y2).to(device)
            S_Ind_torch = torch.argsort(XY).to(device)
            XY_sorted = XY[S_Ind_torch]
            ar_x_sorted = ar_x[S_Ind_torch]
            ar_y_sorted = ar_y[S_Ind_torch]
        else:
            ar_x = np.tile([X2], (width * height_data * color, 1))
            ar_x = torch.tensor(ar_x).to(device)
            ar_x = torch.transpose(ar_x, 1, 0)

            ar_y = np.tile([Y2], (width * height_data * color, 1))
            ar_y = torch.tensor(ar_y).to(device)
            ar_y = torch.transpose(ar_y, 1, 0)

            S_Ind_torch = torch.argsort(XY, dim=0).to(device)
            XY_sorted = torch.gather(XY, 0, S_Ind_torch)
            ar_x_sorted = torch.gather(ar_x, 0, S_Ind_torch)
            ar_y_sorted = torch.gather(ar_y, 0, S_Ind_torch)

        power = 1

        E_CDF = torch.cumsum(ar_x_sorted[:-1], dim=0)
        F_CDF = torch.cumsum(ar_y_sorted[:-1], dim=0)
        height_wd = abs(F_CDF - E_CDF)
        compared = torch.eq(XY_sorted[:-1], XY_sorted[1:])
        height_wd[compared] = 0
        Res = height_wd ** power
        Res = torch.sum(Res, dim=0)

        return Res
    
    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        cvm = self._compute_distance_gpu(data_1, data_2, width, height, color, device)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = torch.cat((data_1, data_2)).to(device)
        reps = 0
        bigger = torch.zeros([width * height * color])
        for ii in tqdm(range(1, bootstrap_samples)):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)

            if len(comb.shape) > 1:
                xa = comb[e, :]
                xb = comb[f, :]
            else:
                xa = comb[e]
                xb = comb[f]

            boost_cvm = self._compute_distance_gpu(xa, xb, width, height, color, device)
            compare = boost_cvm > cvm
            bigger[compare.nonzero()] += 1

        pVal = bigger / bootstrap_samples

        return pVal, cvm


class KuiperDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.KUIPER_DISTANCE

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = np.concatenate([data_1, data_2])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        up = 0
        down = 0
        height = 0
        Res = 0
        E_CDF = 0
        F_CDF = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            if XY_Sorted[ii + 1] != XY_Sorted[ii]:
                height = F_CDF - E_CDF
            if height > up:
                up = height
            if height < down:
                down = height

        K_Dist = abs(down) ** power + abs(up) ** power

        return K_Dist

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        KD = self._compute_distance(data_1, data_2)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = np.concatenate([data_1, data_2])
        reps = 0
        bigger = 0
        for ii in range(1, bootstrap_samples):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            boost_KD = self._compute_distance(comb[e], comb[f])
            if (boost_KD > KD):
                bigger = 1 + bigger

        pVal = bigger / bootstrap_samples

        return pVal, KD
    
    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height_data: int,
                              color: int, device: str) -> torch.Tensor:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = torch.cat([data_1,data_2]).to(device)
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])

        up = 0
        down = 0
        height = 0
        Res = 0
        E_CDF = 0
        F_CDF = 0
        power = 1
        
        if width == height_data == color == 1:
            ar_x = torch.tensor(X2).to(device)
            ar_y = torch.tensor(Y2).to(device)
            S_Ind_torch = torch.argsort(XY).to(device)
            XY_sorted = XY[S_Ind_torch]
            ar_x_sorted = ar_x[S_Ind_torch]
            ar_y_sorted = ar_y[S_Ind_torch]
        else:
            ar_x = np.tile([X2], (width*height_data*color, 1))
            ar_x = torch.tensor(ar_x).to(device)
            ar_x = torch.transpose(ar_x,1,0)

            ar_y = np.tile([Y2], (width*height_data*color, 1))
            ar_y = torch.tensor(ar_y).to(device)
            ar_y = torch.transpose(ar_y,1,0)

            S_Ind_torch = torch.argsort(XY, dim=0).to(device)
            XY_sorted = torch.gather(XY, 0, S_Ind_torch)
            ar_x_sorted = torch.gather(ar_x, 0, S_Ind_torch)
            ar_y_sorted = torch.gather(ar_y, 0, S_Ind_torch)

        E_CDF = torch.cumsum(ar_x_sorted[:-1], dim=0)
        F_CDF = torch.cumsum(ar_y_sorted[:-1], dim=0)
        height = F_CDF-E_CDF
        up = torch.max(height, dim=0)
        down = torch.min(height, dim=0)
        K_Dist = torch.pow(torch.abs(down.values), power) + torch.pow(torch.abs(up.values), power)

        return K_Dist

    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        KD = self._compute_distance_gpu(data_1, data_2, width, height, color, device)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = torch.cat((data_1,data_2)).to(device)
        reps = 0
        bigger = torch.zeros([width*height*color])
        for ii in tqdm(range(1, bootstrap_samples)):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            if len(comb.shape) > 1:
                xa = comb[e, :]
                xb = comb[f, :]
            else:
                xa = comb[e]
                xb = comb[f]
            boost_WD = self._compute_distance_gpu(xa, xb, width, height, color, device)
            compare = boost_WD > KD
            bigger[compare.nonzero()] += 1

        pVal = bigger/bootstrap_samples
        
        return pVal, KD


class AndersonDarlingDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.ANDERSON_DARLING_DISTANCE

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = np.concatenate([data_1, data_2])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res = 0
        E_CDF = 0
        F_CDF = 0
        G_CDF = 0
        height = 0
        SD = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            G_CDF = G_CDF + 1 / n
            SD = (n * G_CDF * (1 - G_CDF)) ** 0.5
            height = abs(F_CDF - E_CDF)
            if XY_Sorted[ii + 1] != XY_Sorted[ii]:
                if SD > 0:
                    Res = Res + (height / SD) ** power

        AD_Dist = Res

        return AD_Dist

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        AD = self._compute_distance(data_1, data_2)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = np.concatenate([data_1, data_2])
        reps = 0
        bigger = 0
        for ii in range(1, bootstrap_samples):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            boost_AD = self._compute_distance(comb[e], comb[f])
            if (boost_AD > AD):
                bigger = 1 + bigger

        pVal = bigger / bootstrap_samples

        return pVal, AD
    
    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height_data: int,
                              color: int, device: str) -> torch.Tensor:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = torch.cat([data_1, data_2]).to(device)
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        if width == height_data == color == 1:
            ar_x = torch.tensor(X2).to(device)
            ar_y = torch.tensor(Y2).to(device)
            S_Ind_torch = torch.argsort(XY).to(device)
            XY_sorted = XY[S_Ind_torch]
            ar_x_sorted = ar_x[S_Ind_torch]
            ar_y_sorted = ar_y[S_Ind_torch]
        else:
            ar_x = np.tile([X2], (width * height_data * color, 1))
            ar_x = torch.tensor(ar_x).to(device)
            ar_x = torch.transpose(ar_x, 1, 0)

            ar_y = np.tile([Y2], (width * height_data * color, 1))
            ar_y = torch.tensor(ar_y).to(device)
            ar_y = torch.transpose(ar_y, 1, 0)

            S_Ind_torch = torch.argsort(XY, dim=0).to(device)
            XY_sorted = torch.gather(XY, 0, S_Ind_torch)
            ar_x_sorted = torch.gather(ar_x, 0, S_Ind_torch)
            ar_y_sorted = torch.gather(ar_y, 0, S_Ind_torch)

        power = 1

        E_CDF = torch.cumsum(ar_x_sorted[:-1], dim=0)
        F_CDF = torch.cumsum(ar_y_sorted[:-1], dim=0)
        G_CDF = torch.arange(1, n) * (1/n)
        height_wd = abs(F_CDF - E_CDF)
        SD = (n * G_CDF * (1-G_CDF))**0.5
        SD = SD.to(device)
        compared = torch.eq(XY_sorted[:-1], XY_sorted[1:])
        height_wd[compared] = 0
        if len(height_wd.shape) == 1:
            Res = (height_wd / SD) ** power
        else:
            Res = (height_wd / SD[:, None]) ** power
        Res = torch.sum(Res, dim=0)

        return Res
    
    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        WD = self._compute_distance_gpu(data_1, data_2, width, height, color, device)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = torch.cat((data_1, data_2)).to(device)
        reps = 0
        bigger = torch.zeros([width * height * color])
        for ii in tqdm(range(1, bootstrap_samples)):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)

            if len(comb.shape) > 1:
                xa = comb[e, :]
                xb = comb[f, :]
            else:
                xa = comb[e]
                xb = comb[f]

            boost_WD = self._compute_distance_gpu(xa, xb, width, height, color, device)
            compare = boost_WD > WD
            bigger[compare.nonzero()] += 1

        pVal = bigger / bootstrap_samples

        return pVal, WD


class KolmogorovSmirnovDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.KOLMOGOROV_SMIRNOV_DISTANCE

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = np.concatenate([data_1, data_2])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res = 0
        height = 0
        E_CDF = 0
        F_CDF = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            if XY_Sorted[ii + 1] != XY_Sorted[ii]:
                height = abs(F_CDF - E_CDF)
            if height > Res:
                Res = height

        KS_Dist = Res ** power

        return KS_Dist

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        KSD = self._compute_distance(data_1, data_2)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = np.concatenate([data_1, data_2])
        reps = 0
        bigger = 0
        for ii in range(1, bootstrap_samples):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            boost_KSD = self._compute_distance(comb[e], comb[f])
            if (boost_KSD > KSD):
                bigger = 1 + bigger

        pVal = bigger / bootstrap_samples

        return pVal, KSD
    
    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height_data: int,
                              color: int, device: str) -> torch.Tensor:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = torch.cat([data_1, data_2]).to(device)
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        if width == height_data == color == 1:
            ar_x = torch.tensor(X2).to(device)
            ar_y = torch.tensor(Y2).to(device)
            S_Ind_torch = torch.argsort(XY).to(device)
            XY_sorted = XY[S_Ind_torch]
            ar_x_sorted = ar_x[S_Ind_torch]
            ar_y_sorted = ar_y[S_Ind_torch]
        else:
            ar_x = np.tile([X2], (width * height_data * color, 1))
            ar_x = torch.tensor(ar_x).to(device)
            ar_x = torch.transpose(ar_x, 1, 0)

            ar_y = np.tile([Y2], (width * height_data * color, 1))
            ar_y = torch.tensor(ar_y).to(device)
            ar_y = torch.transpose(ar_y, 1, 0)

            S_Ind_torch = torch.argsort(XY, dim=0).to(device)
            XY_sorted = torch.gather(XY, 0, S_Ind_torch)
            ar_x_sorted = torch.gather(ar_x, 0, S_Ind_torch)
            ar_y_sorted = torch.gather(ar_y, 0, S_Ind_torch)

        power = 1

        E_CDF = torch.cumsum(ar_x_sorted[:-1], dim=0)
        F_CDF = torch.cumsum(ar_y_sorted[:-1], dim=0)
        height_wd = abs(F_CDF - E_CDF)
        compared = torch.eq(XY_sorted[:-1], XY_sorted[1:])
        height_wd[compared] = 0
        Res = torch.max(height_wd, dim=0)
        
        KS_Dist = Res.values**power

        return KS_Dist
    
    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        WD = self._compute_distance_gpu(data_1, data_2, width, height, color, device)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = torch.cat((data_1, data_2)).to(device)
        reps = 0
        bigger = torch.zeros([width * height * color])
        for ii in tqdm(range(1, bootstrap_samples)):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)

            if len(comb.shape) > 1:
                xa = comb[e, :]
                xb = comb[f, :]
            else:
                xa = comb[e]
                xb = comb[f]

            boost_WD = self._compute_distance_gpu(xa, xb, width, height, color, device)
            compare = boost_WD > WD
            bigger[compare.nonzero()] += 1

        pVal = bigger / bootstrap_samples

        return pVal, WD


class DTSDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.DTS_DISTANCE

    def _compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = np.concatenate([data_1, data_2])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res = 0
        E_CDF = 0
        F_CDF = 0
        G_CDF = 0
        hight = 0
        width = 0
        power = 1

        for ii in range(0, n - 2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            G_CDF = G_CDF + 1 / n
            SD = (n * G_CDF * (1 - G_CDF)) ** 0.5
            height = abs(F_CDF - E_CDF)
            width = XY_Sorted[ii + 1] - XY_Sorted[ii]
            if SD > 0:
                Res = Res + ((height / SD) ** power) * width

        DTS_D = Res

        return DTS_D

    def _compute_distance_p_value(self, data_1: np.ndarray, data_2: np.ndarray, bootstrap_samples: int = 1000
                                  ) -> Tuple[float, float]:
        DTS = self._compute_distance(data_1, data_2)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = np.concatenate([data_1, data_2])
        reps = 0
        bigger = 0
        for ii in range(1, bootstrap_samples):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)
            boost_DTS = self._compute_distance(comb[e], comb[f])
            if (boost_DTS > DTS):
                bigger = 1 + bigger

        pVal = bigger / bootstrap_samples

        return pVal, DTS
    
    def _compute_distance_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height_data: int,
                              color: int, device: str) -> torch.Tensor:
        nx = len(data_1)
        ny = len(data_2)
        n = nx + ny

        XY = torch.cat([data_1, data_2]).to(device)
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        if width == height_data == color == 1:
            ar_x = torch.tensor(X2).to(device)
            ar_y = torch.tensor(Y2).to(device)
            S_Ind_torch = torch.argsort(XY).to(device)
            XY_sorted = XY[S_Ind_torch]
            ar_x_sorted = ar_x[S_Ind_torch]
            ar_y_sorted = ar_y[S_Ind_torch]
        else:
            ar_x = np.tile([X2], (width * height_data * color, 1))
            ar_x = torch.tensor(ar_x).to(device)
            ar_x = torch.transpose(ar_x, 1, 0)

            ar_y = np.tile([Y2], (width * height_data * color, 1))
            ar_y = torch.tensor(ar_y).to(device)
            ar_y = torch.transpose(ar_y, 1, 0)

            S_Ind_torch = torch.argsort(XY, dim=0).to(device)
            XY_sorted = torch.gather(XY, 0, S_Ind_torch)
            ar_x_sorted = torch.gather(ar_x, 0, S_Ind_torch)
            ar_y_sorted = torch.gather(ar_y, 0, S_Ind_torch)

        power = 1

        E_CDF = torch.cumsum(ar_x_sorted[:-1], dim=0)
        F_CDF = torch.cumsum(ar_y_sorted[:-1], dim=0)
        G_CDF = torch.arange(1, n) * (1/n)
        height_wd = abs(F_CDF - E_CDF)
        SD = (n * G_CDF * (1-G_CDF))**0.5
        SD = SD.to(device)
        width_wd = XY_sorted[1:] - XY_sorted[:-1]
        if len(height_wd.shape) == 1:
            Res = ( (height_wd / SD) ** power ) 
            Res = Res * width_wd
        else:
            Res = ( (height_wd / SD[:, None]) ** power)
            Res = Res * width_wd
        
        Res[ ~(SD > 0) ] = 0
        Res = torch.sum(Res, dim=0)
        
        return Res
    
    def _compute_distance_p_value_gpu(self, data_1: torch.Tensor, data_2: torch.Tensor, width: int, height: int,
                                      color: int, device: str, bootstrap_samples: int
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        WD = self._compute_distance_gpu(data_1, data_2, width, height, color, device)
        na = len(data_1)
        nb = len(data_2)
        n = na + nb
        comb = torch.cat((data_1, data_2)).to(device)
        reps = 0
        bigger = torch.zeros([width * height * color])
        for ii in tqdm(range(1, bootstrap_samples)):
            e = random.sample(range(n), na)
            f = random.sample(range(n), nb)

            if len(comb.shape) > 1:
                xa = comb[e, :]
                xb = comb[f, :]
            else:
                xa = comb[e]
                xb = comb[f]

            boost_WD = self._compute_distance_gpu(xa, xb, width, height, color, device)
            compare = boost_WD > WD
            bigger[compare.nonzero()] += 1

        pVal = bigger / bootstrap_samples

        return pVal, WD
