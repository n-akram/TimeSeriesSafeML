from enum import Enum
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
    

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
        """This function returns the distance between the two given
        numpy arrays.

        :param data_1: flattened numpy ndarray 
        :param data_2: flattened numpy ndarray 
        :return: distance measures.
        """
        raise NotImplementedError
        

class WassersteinDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.WASSERSTEIN_DISTANCE

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
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

    
class CramerVonMisesDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.CRAMER_VON_MISES_DISTANCE

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
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

    
class KuiperDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.KUIPER_DISTANCE

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
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

    
class AndersonDarlingDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.ANDERSON_DARLING_DISTANCE

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
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

    
class KolmogorovSmirnovDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.KOLMOGOROV_SMIRNOV_DISTANCE

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
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


class DTSDistance(ECDFDistanceMeasure):
    DISTANCE_TYPE = ECDFDistanceMeasures.DTS_DISTANCE

    def compute_distance(self, data_1: np.ndarray, data_2: np.ndarray) -> float:
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