import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import lmfit
import math

def deriv(y,t, alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D):
    alfa_As = [alfa_AA, alfa_AB, alfa_AC, alfa_AD]
    alfa_Bs = [alfa_BA, alfa_BB, alfa_BC, alfa_BD]
    alfa_Cs = [alfa_CA, alfa_CB, alfa_CC, alfa_CD]
    alfa_Ds = [alfa_DA, alfa_DB, alfa_DC, alfa_DD]
    eta_AB, eta_BC, eta_CD = [1 / 48, 1 / 80, 1 / 80]
    birth, death = 1.06638875, 1 / 78.4
    SA,IA,SB,IB,SC,IC,SD,ID = y
    Is = np.array([IA,IB,IC,ID])
    dSAdt = - (SA / (SA + IA)) * (sum(alfa_As * Is)) + beta_A * IA - eta_AB * SA + birth
    dIAdt = (SA / (SA + IA)) * (sum(alfa_As * Is)) - beta_A * IA - eta_AB * IA
    dSBdt = - (SB / (SB + IB)) * (sum(alfa_Bs * Is)) + beta_B * IB - eta_BC * SB + eta_AB * SA
    dIBdt = (SB / (SB + IB)) * (sum(alfa_Bs * Is)) - beta_B * IB - eta_BC * IB + eta_AB * IA
    dSCdt = - (SC / (SC + IC)) * (sum(alfa_Cs * Is)) + beta_C * IC - eta_CD * SC + eta_BC * SB
    dICdt = (SC / (SC + IC)) * (sum(alfa_Cs * Is)) - beta_C * IC - eta_CD * IC + eta_BC * IB
    dSDdt = - (SD / (SD + ID)) * (sum(alfa_Ds * Is)) + beta_D * ID + eta_CD * SC - death * SD
    dIDdt = (SD / (SD + ID)) * (sum(alfa_Ds * Is)) - beta_D * ID + eta_CD * IC - death * ID
    return dSAdt,dIAdt,dSBdt,dIBdt,dSCdt,dICdt,dSDdt,dIDdt

def Model(T_end,alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D,):
    y0 = 49.937-GroupA[0], GroupA[0], 85.323-GroupB[0], GroupB[0], 73.513-GroupC[0], GroupC[0], 40.760-GroupD[0], GroupD[0]
    t = np.linspace(0,T_end, num=T_end+1)
    ret = odeint(deriv,y0,t,args=(alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D))
    SA,IA,SB,IB,SC,IC,SD,ID = ret.T
    return t, SA,IA,SB,IB,SC,IC,SD,ID

def fitter(x,alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D):
    # print([alfa_AA, alfa_AB, alfa_AC, alfa_AD], [alfa_BA, alfa_BB, alfa_BC, alfa_BD],[alfa_CA, alfa_CB, alfa_CC, alfa_CD], [alfa_DA, alfa_DB, alfa_DC, alfa_DD], [beta_A,beta_B,beta_C,beta_D])
    t, SA,IA,SB,IB,SC,IC,SD,ID = Model(T_end,alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D)
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
params_min_max = {"alfa_AA": (0.14499257406962618,0,2),#0.14499257406962618
                  "alfa_AB": (0,0,2),
                  "alfa_AC": (0,0,2),
                  "alfa_AD": (0,0,2),
                  "alfa_BA": (0,0,2),
                  "alfa_BB": (0.17116438367248832,0,2),#0.17116438367248832
                  "alfa_BC": (0,0,2),
                  "alfa_BD": (0,0,2),
                  "alfa_CA": (0,0,2),
                  "alfa_CB": (0,0,2),
                  "alfa_CC": (0.042058926760615734,0,2),#0.042058926760615734
                  "alfa_CD": (0,0,2),
                  "alfa_DA": (0,0,2),
                  "alfa_DB": (0,0,2),
                  "alfa_DC": (0,0,2),
                  "alfa_DD": (0.04607447725203018,0,2),#0.04607447725203018
                  "beta_A": (4.0021447267335475e-10,0,1), #4.0021447267335475e-10
                  "beta_B": (0.028982012452169026,0,1), #0.028982012452169026
                  "beta_C": (0.004857711928547059,0,1), #0.004857711928547059
                  "beta_D": (0.02783187151432015,0,1)} #0.02783187151432015
mod = lmfit.Model(fitter)
for kwarg, (init, mini, maxi) in params_min_max.items():
    mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)
params = mod.make_params()

print('fitting now...')
X = np.array(list(range(4*len(GroupA))))
result = mod.fit(Groups,params,method="leastsq", x = X)
print('fitted')
MSE = sum(result.residual**2)/len(result.residual)
print("MSE = ", MSE)
parameters = []
for key in result.best_values.keys():
    parameters.append(result.best_values[key])
print(parameters)
alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D = parameters


# alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D = [0,0,0,0.147,0.258,0,0,0,0,0.681,0.0008,0,0,0,0,1.380,0,0.039,0.747,1]
T_end=len(GroupA)+5*4
t, SA,IA,SB,IB,SC,IC,SD,ID = Model(T_end,alfa_AA, alfa_AB, alfa_AC, alfa_AD, alfa_BA, alfa_BB, alfa_BC, alfa_BD, alfa_CA, alfa_CB, alfa_CC, alfa_CD, alfa_DA, alfa_DB, alfa_DC, alfa_DD, beta_A,beta_B,beta_C,beta_D)
plt.plot(IA,color='blue')
plt.plot(IB,color='red')
plt.plot(IC,color='green')
plt.plot(ID,color='purple')
plt.plot(GroupA,color='blue') #I_A
plt.plot(GroupB,color='red') #I_B
plt.plot(GroupC,color='green') #I_C
plt.plot(GroupD,color='purple') #I_D
plt.legend(["13-24", "25-44", "45-64","65+"])
plt.show()