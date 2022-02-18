import numpy as np
import math
import matplotlib.pyplot as plt

def fit_polynomial_curve_to_data_and_visualize(x, y, degree=2, interpolation=100, xlabel='Distance', ylabel='Performance'):

    z = np.polyfit(x, y, degree)

    fit = np.poly1d(z)

    x_min = x.min()
    x_max = x.max()

    xp = np.linspace(x_min, x_max, interpolation)
    yp = fit(xp)

    #Plotting
    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.subplots()
    ax1.plot(xp, yp, color = 'r',alpha = 0.5, label = 'Polynomial fit')
    ax1.scatter(x, y, s = 5, color = 'b', label = 'Data points')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title('Polynomial fit for given (x,y) data points')
    ax1.legend()
    plt.show()

def compute_ratio_from_curve_fit(x, y, performance, degree=2):

    z = np.polyfit(x, y, degree)

    fit = np.poly1d(z)     

    a = fit.coefficients[0]
    b = fit.coefficients[1]
    c = fit.coefficients[2]

    WDFromCurve_1 = (-b +  math.sqrt(b**2 - 4 * a * (c - performance))) / (2 * a)
    WDFromCurve_2 = (-b -  math.sqrt(b**2 - 4 * a * (c - performance))) / (2 * a)

    WDFromCurve_1 = np.round(WDFromCurve_1, 4)
    WDFromCurve_2 = np.round(WDFromCurve_2, 4)

    return WDFromCurve_1, WDFromCurve_2

