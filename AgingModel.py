
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import lmfit
import math

def deriv(y,t, alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D):
    eta_AB, eta_BC, eta_CD = [1 / 48, 1 / 80, 1 / 80]
    birth, death = 1.06638875, 1 / 78.4
    SA,IA,SB,IB,SC,IC,SD,ID = y
    dSAdt = - (SA / (SA + IA)) * (IA*alfa_A) + beta_A * IA - eta_AB * SA + birth
    dIAdt = (SA / (SA + IA)) * (IA*alfa_A) - beta_A * IA - eta_AB * IA
    dSBdt = - (SB / (SB + IB)) * (IB*alfa_B) + beta_B * IB - eta_BC * SB + eta_AB * SA
    dIBdt = (SB / (SB + IB)) * (IB*alfa_B) - beta_B * IB - eta_BC * IB + eta_AB * IA
    dSCdt = - (SC / (SC + IC)) * (IC*alfa_C) + beta_C * IC - eta_CD * SC + eta_BC * SB
    dICdt = (SC / (SC + IC)) * (IC*alfa_C) - beta_C * IC - eta_CD * IC + eta_BC * IB
    dSDdt = - (SD / (SD + ID)) * (ID*alfa_D) + beta_D * ID + eta_CD * SC - death * SD
    dIDdt = (SD / (SD + ID)) * (ID*alfa_D) - beta_D * ID + eta_CD * IC - death * ID
    return dSAdt,dIAdt,dSBdt,dIBdt,dSCdt,dICdt,dSDdt,dIDdt

def Model(T_end,alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D):
    y0 = 49.937-GroupA[0], GroupA[0], 85.323-GroupB[0], GroupB[0], 73.513-GroupC[0], GroupC[0], 40.760-GroupD[0], GroupD[0]
    t = np.linspace(0,T_end, num=T_end+1)
    ret = odeint(deriv,y0,t,args=(alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D))
    SA,IA,SB,IB,SC,IC,SD,ID = ret.T
    return t, SA,IA,SB,IB,SC,IC,SD,ID

def fitter(x,alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D):
    t, SA,IA,SB,IB,SC,IC,SD,ID = Model(T_end,alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D)
    SUM = np.concatenate((IA,IB,IC,ID))
    return SUM[x]

print("starting")
Data = pd.read_excel("FacebookUS.xlsx")
GroupA = 0.711*np.array(Data["13-24"])
GroupB = 0.591*np.array(Data["25-44"])
GroupC = 0.846*np.array(Data["45-64"])
GroupD = 0.755*np.array(Data["65+"])
Groups = np.concatenate((GroupA,GroupB,GroupC,GroupD))
print(Groups)
T_end = len(GroupA)-1
params_min_max = {"alfa_A": (0,0,2),
                  "alfa_B": (0,0,2),
                  "alfa_C": (0,0,2),
                  "alfa_D": (0,0,2),
                  "beta_A": (0,0,1),
                  "beta_B": (0,0,1),
                  "beta_C": (0,0,1),
                  "beta_D": (0,0,1)}
mod = lmfit.Model(fitter)
for kwarg, (init, mini, maxi) in params_min_max.items():
    mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)
params = mod.make_params()

print('fitting now...')
X = np.array(list(range(4*len(GroupA))))
result = mod.fit(Groups,params,method="leastsq", x = X)
# result.plot_fit()
print('fitted')
MSE = sum(result.residual**2)/len(result.residual)
print("MSE = ", MSE)
parameters = []
for key in result.best_values.keys():
    parameters.append(result.best_values[key])
print(parameters)
alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D= parameters


T_end=len(GroupA)+5*4
t, SA,IA,SB,IB,SC,IC,SD,ID = Model(T_end,alfa_A, alfa_B,alfa_C, alfa_D, beta_A,beta_B,beta_C,beta_D)
plt.plot(IA,color='blue')
plt.plot(IB,color='red')
plt.plot(IC,color='green')
plt.plot(ID,color='purple')
plt.plot(GroupA,color='blue') #I_A
plt.plot(GroupB,color='red') #I_B
plt.plot(GroupC,color='green') #I_C
plt.plot(GroupD,color='purple') #I_D
plt.legend(["13-24", "25-44", "45-64","65+"])
plt.xlabel("Quarters")
plt.ylabel("Users(million)")
plt.show()

# percentages:
N = IA+IB+IC+ID
plt.plot(IA/N,color='blue')
plt.plot(IB/N,color='red')
plt.plot(IC/N,color='green')
plt.plot(ID/N,color='purple')
plt.legend(["13-24", "25-44", "45-64","65+"])
plt.xlabel("Quarters")
plt.ylabel("Fraction of users(%)")
plt.axvline(len(GroupA)-1,color="red",linestyle='dotted')
plt.text(len(GroupA)-1+0.5, min(ID/N) , 'now', color='red', ha='left', va='top')
plt.show()
