import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def f(X,params):
    alfa_A, beta_A, alfa_B, beta_B, alfa_C, beta_C, alfa_D, beta_D, eta_AB, eta_BC, eta_CD, birth,death = params
    S_A0,I_A0,S_B0,I_B0,S_C0,I_C0,S_D0,I_D0 = X
    S_A = - alfa_A * S_A0 * I_A0 /(S_A0+I_A0) + beta_A * I_A0 - eta_AB*S_A0 + birth
    I_A = alfa_A * S_A0 * I_A0 /(S_A0+I_A0) - beta_A * I_A0 - eta_AB*I_A0
    S_B = - alfa_B * S_B0 * I_B0 / (S_B0+I_B0) + beta_B * I_B0 - eta_BC*S_B0 + eta_AB*S_A0
    I_B = alfa_B * S_B0 * I_B0 / (S_B0+I_B0) - beta_B * I_B0 - eta_BC*I_B0 + eta_AB*I_A0
    S_C = - alfa_C * S_C0 * I_C0 /(S_C0+I_C0) + beta_C * I_C0 - eta_CD*S_C0 + eta_BC*S_B0
    I_C = alfa_C * S_C0 * I_C0 /(S_C0+I_C0) - beta_C * I_C0 - eta_CD*I_C0 + eta_BC*I_B0
    S_D = - alfa_D * S_D0 * I_D0 /(S_D0+I_D0) + beta_D * I_D0 + eta_CD*S_C0 - death*S_D0
    I_D = alfa_D * S_D0 * I_D0 /(S_D0+I_D0) - beta_D * I_D0 + eta_CD*I_C0 - death*I_D0
    return np.array([S_A,I_A,S_B,I_B,S_C,I_C,S_D,I_D])


def RK4(X, params,dt):
    k1 = dt*f(X, params)
    k2 = dt*f(X+ 0.5*k1, params)
    k3 = dt*f(X+ 0.5*k2, params)
    k4 = dt*f(X+k3, params)
    X_new = X + (k1+2*k2+2*k3+k4)/6
    return X_new

def Model(T_end,dt,params,startingValues):
    S_A,I_A,S_B,I_B,S_C,I_C,S_D,I_D = startingValues
    time=[0]
    X = np.array([S_A,I_A,S_B,I_B,S_C,I_C,S_D,I_D])
    A_list = [I_A]
    B_list = [I_B]
    C_list = [I_C]
    D_list = [I_D]
    while time[-1] < T_end:
        time.append(time[-1] + dt)
        X = RK4(X, params,dt)
        S_A, I_A, S_B, I_B, S_C, I_C, S_D, I_D = X
        A_list.append(I_A)
        B_list.append(I_B)
        C_list.append(I_C)
        D_list.append(I_D)
    return time, [A_list,B_list,C_list,D_list]

def fit(Groups,T_end,dt,startingValues,amount):
    endpoints = [0,2,0,1,0,2,0,1,0,2,0,1,0,2,0,1] # all alfa en beta ranges
    minparams = [0,0,0,0,0,0,0,0,1/48,1/80,1/80,1.06638875,1/78.4] #[alfaA,betaA,alfaB,betaB,alfaC,betaC,alfaD,betaD,etaAB,etaBC,etaCD,birth,death]
    minerror = 10**10
    while max([endpoints[2*k+1]-endpoints[2*k] for k in range(4)]) > 10**(-3):
        print(max([endpoints[2*k+1]-endpoints[2*k] for k in range(4)]))
        minparams, minerror = helper(amount, minparams, minerror, endpoints, T_end, dt,Groups, startingValues)
        for k in range(len(endpoints)//2):
            if minparams[k] < (endpoints[2*k+1]+endpoints[2*k])/2:
                endpoints[2*k+1] = (endpoints[2*k+1]+endpoints[2*k])/2
            else:
                endpoints[2*k] = (endpoints[2*k+1] + endpoints[2*k]) / 2
    return minparams, minerror

def helper(amount, minparams, minerror, endpoints, T_end, dt,Groups, startingValues):
    birth = 1.06638875
    death = 1/78.4
    eta_AB = 1/48
    eta_BC = 1/80
    eta_CD = 1/80
    ranges = []
    alfasA = np.linspace(endpoints[0], endpoints[1], num=amount)
    betasA = np.linspace(endpoints[2], endpoints[3], num=amount)
    alfasB = np.linspace(endpoints[4], endpoints[5], num=amount)
    betasB = np.linspace(endpoints[6], endpoints[7], num=amount)
    alfasC = np.linspace(endpoints[8], endpoints[9], num=amount)
    betasC = np.linspace(endpoints[10], endpoints[11], num=amount)
    alfasD = np.linspace(endpoints[12], endpoints[13], num=amount)
    betasD = np.linspace(endpoints[14], endpoints[15], num=amount)
    for alfa_A in alfasA:
        for beta_A in betasA:
            for alfa_B in alfasB:
                for beta_B in betasB:
                    for alfa_C in alfasC:
                        for beta_C in betasC:
                            for alfa_D in alfasD:
                                for beta_D in betasD:
                                    params = [alfa_A, beta_A, alfa_B, beta_B, alfa_C, beta_C, alfa_D, beta_D, eta_AB, eta_BC, eta_CD, birth,death]
                                    time, Lists = Model(T_end, dt, params, startingValues)
                                    e = Error(Lists,Groups)
                                    if e < minerror:
                                        minparams = params
                                        minerror = e
    return minparams, minerror

def Error(Lists, Groups):
    MSE = 0
    for k in range(len(Lists)):
        n = len(Groups[k])
        for i in range(n):
            MSE += (Groups[k][i] - Lists[k][i]) ** 2 / n
    return MSE

Data = pd.read_excel("FacebookUS.xlsx")
GroupA = Data["13-24"]
GroupB = Data["25-44"]
GroupC = Data["45-64"]
GroupD = Data["65+"]
Groups = [GroupA,GroupB,GroupC,GroupD]



#
T_end = 50
dt = 1
# params = [0.28125, 0.125, 0.15559895833333334, 0.078125, 1.03515625, 0.7854817708333334, 0.020182291666666668, 0.015299479166666666, 0.020833333333333332, 0.0125, 0.0125, 1.06638875, 0.012755102040816325]
startingValues = [100, GroupA[0], 200, GroupB[0], 200, GroupC[0], 100, GroupD[0]]
params, error = fit(Groups,T_end,dt,startingValues,2)
print(params)
#
T_end=200
time,Lists = Model(T_end,dt,params,startingValues)
print(Error(Lists,Groups))
plt.plot(time,Lists[0],color='blue') #I_A
plt.plot(time,Lists[1],color='red') #I_B
plt.plot(time,Lists[2],color='green') #I_C
plt.plot(time,Lists[3],color='purple') #I_D

plt.plot(Groups[0],color='blue') #I_A
plt.plot(Groups[1],color='red') #I_B
plt.plot(Groups[2],color='green') #I_C
plt.plot(Groups[3],color='purple') #I_D
plt.show()

