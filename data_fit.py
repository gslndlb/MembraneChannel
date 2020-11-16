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

e_bonds = [0.6, 2, 15, 20, 40]
name_bonds = ["Keesom", "Dipole-dipole", "Ion-dipole", "Weak hydrogen", "Strong hydrogen"] 
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
# plt.plot(PC, popt_init[:, 0]*R*T/1e3, label='guess')
plt.plot(PC, -popt_end[:, 0]*R*T/1e3, '--o', label='fit')
plt.ylabel(r'$\Delta E$ (kJ/mol)')
plt.legend()
plt.yscale('log')
plt.subplot(212)
# plt.plot(PC, popt_init[:, 1])
plt.plot(PC, popt_end[:, 1], '--o')
plt.ylabel(r'$\beta A/a^2$')
plt.xlabel('x')
plt.yscale('log')
plt.tight_layout()

plt.figure(3)
for ii in range(len(e_bonds)):
    plt.plot(PC, -popt_end[:, 0]*R*T/1e3/e_bonds[ii], '--o', label=name_bonds[ii], color=colors[ii])
plt.ylabel(r'$n_s(C)$')
plt.xlabel(r'$C$')
plt.yscale('log')
plt.legend()

ii = 1
# plt.close('all')
Cs = np.linspace(16, 20, 100)
alphas = np.linspace(17.5, 20, 10)
alphas = [18.25]
plt.figure(4)
plt.plot(PC, -popt_end[:, 0]*R*T/1e3/e_bonds[ii], '--o', label=name_bonds[ii], color=colors[ii])

def test_fun(C, alpha):
    res = np.zeros(len(C))
    thres_ok = np.zeros(len(C), dtype=bool)

    A = -popt_end[0, 0]*R*T/1e3/e_bonds[ii]/(16/np.sqrt(alpha**2-16**2))

    sqrt_ok = C < alpha
    thres_ok[sqrt_ok] = A*C[sqrt_ok]/np.sqrt(alpha**2 - C[sqrt_ok]**2) < -popt_end[2, 0]*R*T/1e3/e_bonds[ii]
    res[thres_ok] = A*C[thres_ok]/np.sqrt(alpha**2 - C[thres_ok]**2)
    res[~thres_ok] = -popt_end[2, 0]*R*T/1e3/e_bonds[ii]
    return res


# for alpha in alphas:
#     plt.plot(Cs, test_fun(Cs, alpha), color=colors[ii+1])


yy = -popt_end[2, 0]*R*T/1e3/e_bonds[ii]
xx = np.linspace(16, 20, 100)


def fit_fun(C, e_0, alpha, q):
    res = np.zeros(len(C))
    sqrt_ok = C < alpha
    for l in range(1, 5):
        res[sqrt_ok] += e_0*np.minimum(1/np.sqrt(alpha**2 - C[sqrt_ok]**2)/np.sin(np.pi*l/5), q)
        res[~sqrt_ok] += e_0*q
    return res

# popt, pcov = curve_fit(fit_fun, xdata=xx, ydata=yy, p0=(A, alpha, 1.))


A = -popt_end[0, 0]*R*T/1e3/e_bonds[ii]/(16/np.sqrt(alpha**2-16**2))
plt.plot(xx, fit_fun(xx, 6.25, 18.2, 2.1), '', color=colors[ii-1])
plt.xlim(15.8, 20.2)
plt.ylim(0, 60)
