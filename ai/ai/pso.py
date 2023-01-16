# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random as rnd
import  math as mt
import  numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as clrs
def PSOFunction(x):
    return (mt.sin(x)+mt.sin(10*x/3))
def PSOFunction1(x):
    return -(x*x)+5*x+20
def fitnessCalc():
    for i in range(len(values)):
        fitnessArr[i] = PSOFunction(values[i])
def velocityCalc():
    random1 = rnd.uniform(0, 1)
    random2 = rnd.uniform(0, 1)
    for i in range(len(values)):
        velocity[i] = w * velocity[i] + c1 * random1 * (Pbest[i] - values[i]) + c2 * random2 * (Gbest - values[i])
        values[i] += velocity[i]
    if values[i] >10:
        values[i]=10
    if values[i]<-10:
        values[i]=-10
def PbestUpdate():
    for i in range(len(values)):
        if PSOFunction(values[i]) > fitnessArr[i]:
            Pbest[i] = values[i]
start_time= time.perf_counter ()
c1=1
c2=1
w=0.72
iteration = 1
Gbest = 0.0
values = [rnd.uniform(-10,10) for i in range(1000)]
fitnessArr = np.array(np.zeros(1000))
velocity = np.array(np.zeros(1000))
Pbest = list.copy(values)
fitnessCalc()
plt.figure(figsize=(20, 20))
plt.scatter(values, fitnessArr, color = 'red')
plt.title('Parçacık dağılımı ilk fitness')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10,10)
plt.show()
while iteration < 100:
    Gbest = values[fitnessArr.tolist().index(max(fitnessArr))]
    velocityCalc()
    PbestUpdate()
    fitnessCalc()
    iteration += 1
    if 8>Gbest>7.99:
        break
print(Gbest)
plt.figure(figsize=(12, 6))
plt.scatter(values, fitnessArr, color = 'red')
plt.title('Parççacık dağılımı son fitness')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10,10)
plt.show()
end_time = time.perf_counter ()
print(end_time- start_time, "seconds")