# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random as rnd
import  math as mt
import  numpy as np

import matplotlib.pyplot as plt
def PSOFunction(x):
    return mt.sin(x) + mt.sin((10/ 3)*x)
def PSOFunction1(x):
    return -(x*x)+5*x+20
def fitnessCalc():
    for i in range(len(values)):
        fitnessArr[i] = PSOFunction(values[i])
def velocityCalc():
    random1 = rnd.uniform(0, 1)
    random2 = rnd.uniform(0, 1)
    for i in range(len(values)):
        drawDots.append(values[i])
        velocity[i] = w * velocity[i] + c1 * random1 * (Pbest[i] - values[i]) + c2 * random2 * (Gbest - values[i])
        values[i] += velocity[i]
        if values[i]>10:
            values[i]=10
        if values[i]<-10:
            values[i]=-10
def PbestUpdate():
    for i in range(len(values)):
        drawDots.append(values[i])
        if PSOFunction(values[i]) > fitnessArr[i]:
            Pbest[i] = values[i]
c1=1
c2=1
w=1
drawDots=[]
iteration = 1
Gbest = 0.0
values = [rnd.randint(-10,10) for i in range(9)]
fitnessArr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Pbest =list.copy(values)
fitnessCalc()
say=0
while iteration < 200:
    Gbest = values[fitnessArr.index(max(fitnessArr))]
    velocityCalc()
    PbestUpdate()
    fitnessCalc()
    iteration += 1

print(Gbest)

