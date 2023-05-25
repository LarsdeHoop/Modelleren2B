import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def f(X,alfa, beta,N):
    S0,I0 = X
    S = - (alfa * I0 * S0) / N + beta * I0
    I = (alfa * I0 * S0) / N - beta * I0
    return np.array([S,I])


def RK4(S,I, alfa, beta,dt, N):
    X = np.array([S,I])
    k1 = dt*f(X, alfa,beta, N)
    k2 = dt*f(X+ 0.5*k1, alfa,beta, N)
    k3 = dt*f(X+ 0.5*k2, alfa,beta, N)
    k4 = dt*f(X+k3, alfa,beta, N)
    X_new = X + (k1+2*k2+2*k3+k4)/6
    S_new, I_new = X_new
    return S_new, I_new

def Model(T_end,alfa,beta,N,dt,start):
    S = N - start
    I = start
    y = [I]
    x = [0]
    while x[-1] < T_end:
        x.append(x[-1] + dt)
        S, I = RK4(S, I, alfa,beta,dt, N)
        y.append(I)
    return x,y

def fit(qs,MAU,T_end,dt,N):
    amount = 7
    leftA = 252280/5041879
    rightA = 2
    leftS = 200
    rightS = 400
    minparams = (0,0,0)
    minerror = 10**10
    while max(rightA-leftA,rightS-leftS) > 10**(-3):
        print()
        print(leftA, rightA)
        print(leftS, rightS)
        print((rightA-leftA)*(rightS-leftS))
        minparams, minerror = helper(amount, minparams, minerror, leftA,rightA, leftS, rightS,qs,MAU,T_end,dt,N)
        alfa, beta, start = minparams
        if alfa < (rightA+leftA)/2:
            rightA = (rightA+leftA)/2
        else:
            leftA = (rightA+leftA)/2
        if start < (rightS+leftS)/2:
            rightS = (rightS+leftS)/2
        else:
            leftS = (rightS+leftS)/2
    return minparams, minerror

def helper(amount, minparams, minerror, leftA,rightA, leftS, rightS,qs,MAU,T_end,dt,N):
    alfas = np.linspace(leftA, rightA, num=amount)
    starts = np.linspace(leftS, rightS, num=amount)
    for alfa in alfas:
        beta = (3169/4760) * alfa - (53/1591)
        for start in starts:
            #print(alfa, beta, gamma, start)
            x, y = Model(T_end, alfa, beta,N, dt, start)
            if Error(x, y, qs, MAU, dt) < minerror:
                minparams = (alfa, beta, start)
                minerror = Error(x, y, qs, MAU, dt)
    print(minparams)
    return minparams, minerror

def Error(x,y,qs,MAU,dt):
    indexes = int(1/dt) *qs
    n = len(qs)
    MSE = 0
    for k in range(n):
        MSE += (MAU[k]-y[indexes[k]])**2 /n
    return MSE


Active_Users = pd.read_excel("FacebookUsers.xlsx")
MAU = list(Active_Users["MAU"]) # monthly active users

# qs = 'quarterindex for each datapoint' 2006 Q3 => qs=1
qs = []
for i in range(len(Active_Users["Quarter"]) - 1):
    qs.append(11 + i)
qs.insert(0,9)
qs = np.array(qs)
MAU = np.array(MAU)

N = 4760
T_end = 80
dt = 0.01

params, error = fit(qs,MAU,T_end,dt,N)
print(params, error)
alfa,beta,start=params
# alfa,beta,start=0.05765394428099128, 0.0050710946206735386, 325.0

T_end=200
x,y = Model(T_end,alfa,beta,N,dt,start)
plt.plot(x,y, color='blue')
plt.plot(qs,MAU, color='orange')
plt.legend(["fit", "data"], loc="lower right")
plt.xlabel("Quarters since publicly available")
plt.ylabel("Montly active users(million)")
plt.ylim([0,4500])
plt.show()
