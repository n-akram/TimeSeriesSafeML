import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def fit_polynomial_curve_to_data_and_visualize(x, y, degree=2, interpolation=100, xlabel='Distance', ylabel='RMSE'):
    
    """This function fits a poynomail curve for the given (x, y) values and plots the data along with the fitted curve

    :param array x: values for x-axis (independent variable)
    :param array y: values for y-axis (dependent variable)
    :param int degree: degree of polynomial curve, defaults to 2
    :param int interpolation: number of data points for intepolation, defaults to 100
    :param str xlabel: label for x-axis, defaults to 'Distance'
    :param str ylabel: label for y-axis, defaults to 'RMSE'
    """
    
    z = np.polyfit(x, y, degree)

    fit = np.poly1d(z)

    x_min = x.min()
    x_max = x.max()

    xp = np.linspace(x_min, x_max, interpolation)
    yp = fit(xp)

    #Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker_symbol='square', marker = dict(size=10, color='red'), name='Data'))
    fig.add_trace(go.Scatter(x=xp, y=yp, mode='lines', name='Fitted Curve', marker=dict(color='Blue')))
    fig.update_layout(
        title="An illustration of fitting a curve to statistical mapping of dissimilarity and performance",
        xaxis_title="SDD (Wasserstein distance)",
        yaxis_title=f"Performance ({ylabel})",  
        font=dict(
            family="Arial",
            size=16,
            color="Black"
        )
    )
    fig.show()   

def compute_roots_from_curve_fit(x, y, performance, degree=2):

    """This function computes the roots of polynomial equation used to compute ratio for StaDRo

    :return float: Returns roots of the second-degree polynomial equation
    """

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

def sdd_performance_visualization(x, y1, y2, y3, degree=2, interpolation=100, xlabel='Wasserstein distance', model='LSTM', label='Reliance'):

    """This function returns three subplots visualizing the behaviour of SDD with three performance metrics.

    :param array x: SDD (independent variable)
    :param array y1: Performance in RMSE (dependent variable)
    :param array y2: Performance in MAPE (dependent variable)
    :param array y1: Performance in R-Squared (dependent variable)
    :param int degree: degree of polynomial curve, defaults to 2
    :param int interpolation: number of data points for intepolation, defaults to 100
    :param str xlabel: label for x-axis, defaults to 'Wasserstein Distance'
    :param str model: filename extensions to save figures, defaults to 'LSTM'
    """
    
    z1 = np.polyfit(x, y1, degree)
    z2 = np.polyfit(x, y2, degree)
    z3 = np.polyfit(x, y3, degree)

    fit1 = np.poly1d(z1)
    fit2 = np.poly1d(z2)
    fit3 = np.poly1d(z3)

    x_min = x.min()
    x_max = x.max()

    xp = np.linspace(x_min, x_max, interpolation)
    yp1 = fit1(xp)
    yp2 = fit2(xp)
    yp3 = fit3(xp)
    

    #Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.plot(xp, yp1, 'magenta', linestyle='dashed', linewidth=2.5, alpha = 0.6, label = label)
    ax1.scatter(x, y1, 50, 'magenta', "D", label = label)
    ax1.set_xlabel(f'SDD ({xlabel})', fontsize=10, family='monospace')
    ax1.minorticks_on()    
    ax1.tick_params(which='both', width=1, length=7)
    ax1.tick_params(which='minor', length=4)
    ax1.set_ylabel('Root mean square error', fontsize=10, family='monospace')
    ax1.legend(loc='best')

    ax2.plot(xp, yp2, 'magenta', linestyle='dashed', linewidth=2.5, alpha = 0.6, label = label)
    ax2.scatter(x, y2, 50, 'magenta', "D", label = label)
    ax2.set_xlabel(f'SDD ({xlabel})', fontsize=10, family='monospace')
    ax2.minorticks_on()    
    ax2.tick_params(which='both', width=1, length=7)
    ax2.tick_params(which='minor', length=4)
    ax2.set_ylabel('Mean absolute percentage error', fontsize=10, family='monospace')
    ax2.legend(loc='best')

    ax3.plot(xp, yp3, 'magenta', linestyle='dashed', linewidth=2.5, alpha = 0.6, label = label)
    ax3.scatter(x, y3, 50, 'magenta', "D", label = label)
    ax3.set_xlabel(f'SDD ({xlabel})', fontsize=10, family='monospace')
    ax3.minorticks_on()
    ax3.tick_params(which='both', width=1, length=7)
    ax3.tick_params(which='minor', length=4)
    ax3.set_ylabel('R-Squared', fontsize=10, family='monospace')
    ax3.legend(loc='best')
  
    # Use this line of code to save figures
    # plt.savefig(f'{label}_{xlabel}_{model}.eps', dpi=500, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
