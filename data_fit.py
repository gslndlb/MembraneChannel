import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.ion()
plt.close('all')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
HUGE_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)  # fontsize of the figure title

x_data = [[0.715, 10.3, 15., 20.3, 24.4, 30.3],
          [5.01, 10., 14.9, 20.1, 25.2, 30.2, 35.3, 40.1],
          [50.3, 55.1, 60., 65.5]]

y_data = [[0.0164, 0.197, 0.462, 0.519, 0.721, 0.962],
          [0., 0., 0., .0378, 0.118, .4, 0.734, .95],
          [0.00546, .0246, 0.107, .801]]

plt.figure(1)
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
          '#984ea3', '#999999', '#e41a1c', '#dede00']
# Color-blind-friendly set of colors
# colors = ['tab:blue', 'tab:red', 'tab:green']

T = 37 + 273
kb = 8.6e-5
beta = 1/T/kb
R = 8.314


def fit_function(x, Pi, alpha):
    return 1/(1+np.exp(-Pi - alpha*x))


def slopes(xx, yy):
    return (yy[1:] - yy[:-1])/(xx[1:] - xx[:-1])


def threshold(xx, yy):
    return np.argmax(slopes(xx, yy))


def est_alpha(xx, yy):
    return np.mean(slopes(xx, yy))


def est_pi(xx, yy):
    return est_alpha(xx, yy)/yy[threshold(xx, yy)]


fit_function = np.vectorize(fit_function)

popt_init = []
popt_end = []
# for ii in range(x_data.shape[0]):
for ii in range(3):
    xx = np.array(x_data[ii])
    yy = np.array(y_data[ii])
    xxx = np.linspace(np.min(xx)-10, np.max(xx)+10, 100)
    guess = [est_pi(xx, yy), est_alpha(xx, yy)]
    popt, pcov = curve_fit(fit_function, xdata=x_data[ii],
                           ydata=y_data[ii], p0=guess)
    # plt.plot(xx, -np.log(yy)-1, '-o', 'black')
    # print(stats.linregress(xx, np.log(1/yy)-1))
    popt_init.append(guess)
    popt_end.append(popt)
    plt.xlabel('Aspiration pressure (mmHg)')
    plt.ylabel('Opening probability')
    plt.plot(xx, yy, 'o', label='data', color=colors[ii])
    plt.plot(xxx, fit_function(xxx, *popt), ':', label='fit',
             color=colors[ii])
plt.tight_layout()

# print(popt_init, popt_end)
popt_init = np.array(popt_init)
popt_end = np.array(popt_end)
PC = [16, 18, 20]
plt.figure(2)
plt.subplot(211)
plt.plot(PC, popt_init[:, 0]*R*T/1e3, label='guess')
plt.plot(PC, popt_end[:, 0]*R*T/1e3, label='fit')
plt.ylabel(r'$\Delta E$ (kJ/mol)')
plt.legend()
plt.subplot(212)
plt.plot(PC, popt_init[:, 1])
plt.plot(PC, popt_end[:, 1])
plt.ylabel(r'$\beta A/a^2$')
plt.xlabel('x')
plt.tight_layout()
