import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def f(X,alfa, N):
    S0,I0 = X
    S = - (alfa * I0 * S0) / N
    I = (alfa * I0 * S0) / N
    return np.array([S,I])


def RK4(S,I, alfa,dt, N):
    X = np.array([S,I])
    k1 = dt*f(X, alfa,N)
    k2 = dt*f(X+ 0.5*k1, alfa,N)
    k3 = dt*f(X+ 0.5*k2, alfa,N)
    k4 = dt*f(X+k3, alfa,N)
    X_new = X + (k1+2*k2+2*k3+k4)/6
    S_new, I_new = X_new
    return S_new, I_new

def Model(T_end,alfa,N,dt,start):
    S = N - start
    I = start
    y = [I]
    x = [0]
    while x[-1] < T_end:
        x.append(x[-1] + dt)
        S, I = RK4(S, I, alfa,dt, N)
        y.append(I)
    return x,y

def fit(qs,MAU,T_end,dt,N):
    amount = 10
    alfa = 252280/5041879
    leftS = 0
    rightS= 500
    minparams = (alfa,0)
    minerror = 10**10
    while (rightS-leftS) > 10**(-3):
        print()
        print(leftS, rightS)
        print((rightS-leftS))
        minparams, minerror = helper(amount, minparams, minerror, leftS, rightS,qs,MAU,T_end,dt,N)
        alfa, start = minparams
        if start < (rightS+leftS)/2:
            rightS = (rightS+leftS)/2
        else:
            leftS = (rightS+leftS)/2
    return minparams, minerror

def helper(amount, minparams, minerror, leftS, rightS,qs,MAU,T_end,dt,N):
    alfa = minparams[0]
    starts = np.linspace(leftS, rightS, num=amount)
    for start in starts:
        x, y = Model(T_end, alfa, N, dt, start)
        if Error(x, y, qs, MAU, dt) < minerror:
            minparams = (alfa, start)
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
alfa, start = params
# alfa, start = 0.05003690092523046, 327.9942406548394

T_end=100
x,y = Model(T_end,alfa,N,dt,start)
print(Error(x,y,qs,MAU,dt))
plt.plot(x,y, color='blue')
plt.plot(qs,MAU, color='orange')
plt.legend(["fit", "data"], loc="lower right")
plt.xlabel("Quarters since publicly available")
plt.ylabel("Montly active users(million)")
plt.ylim([0,4500])
plt.show()
