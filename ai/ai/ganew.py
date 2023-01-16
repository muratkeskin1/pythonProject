import numpy as np
import random as rnd
import math as mt
from matplotlib import pyplot as plt
import time
import tracemalloc
import  os
import psutil
process = psutil.Process(os.getpid())
iterasyon=100
crosover_rate=0.25
pop_size=20
gen_size=1
def create_chromosome():
    return [rnd.uniform(-10,10) for x in range(0,gen_size)]
def create_initial_population():
    return [create_chromosome() for x in range(0,pop_size)]
def fitness_function(x):
   return mt.sin(x) + mt.sin(10 * x / 3)
def fitness(cr):
    return fitness_function(cr[0])
def probability(fitness_values):
    P=[]
    total=sum(fitness_values)
    for f in fitness_values:
        P.append(f/total)
    return P
def crossover(p1, p2):
    o1 = p1
    o2 =p2
    c=0
    o1[:c] = p2[:c]
    o1[c:] = p1[c:]
    o2[:c] = p1[:c]
    o2[c:] = p2[c:]
    return o1, o2
def mutasyon(mut):
    temp=[]
    temp=mut[:]
    gen=rnd.uniform(-1,1)
    temp[0]+=gen
    if temp[0]>10:
        temp[0]=10
    if temp[0] < -10:
        temp[0] =-10
    return temp
start_time= time.perf_counter ()
tracemalloc.start()
population=create_initial_population()
fitness_values=[]

for c in population:
    fitness_values.append(fitness(c))
epok=0
temppop=[]
fitarr=[]
for i in population:
    temppop.append(i)
for i in range(len(population)):
    fitarr.append(fitness(population[i]))
plt.figure(figsize=(12, 6))
plt.scatter(temppop, fitarr, color = 'red')
plt.title('Kilo Boy Oranı Grafiği2')
plt.xlabel('Kilo')
plt.ylabel('Boy')
plt.xlim(-10,10)
plt.show()
while epok<iterasyon:
    P=probability(fitness_values)
    C=np.cumsum(P)

    rulet_parents=[]
    for i in range(0,len(C)):
        r=rnd.random()
        for j in range(0,len(C)):
            if C[j]>r:
                rulet_parents.append(j)
                break

    crosover_parents=[]
    k=0
    while k<pop_size:
        r=rnd.random()
        if(r<crosover_rate):
            if(rulet_parents[k] not in crosover_parents):
                crosover_parents.append(rulet_parents[k])
        k=k+1


    if(len(crosover_parents)>=2):
        for i in range(0,len(crosover_parents)):
            for j in range (i+1, len(crosover_parents)):
                o1,o2=crossover(population[crosover_parents[i]]
                        ,population[crosover_parents[j]])
                population.append(o1)
                population.append(o2)
                fitness_values.append(fitness(o1))
                fitness_values.append(fitness(o2))
    else:
        print("Crossover icin yetrli birey gelmedi !!!")

    for r in range(pop_size):
        mut=mutasyon(population[rnd.randint(0,len(population)-1)])
        population.append(mut)
        fitness_values.append(fitness(mut))
    zip_list=zip(fitness_values,population)

    sort_list=sorted(zip_list,reverse=False)
    p=len(population)

    while p>pop_size:
        sort_list.pop()
        p=p-1
    population=[]
    fitness_values=[]
    for f,p in list(sort_list):
        population.append(p)
        fitness_values.append(f)
    epok+=1
    if 8>fitness_values[0]>7.99:
        break
print("En iyi birey:",population[0]," fitness:",fitness_values[0])
temppop=[]
fitarr=[]
for i in population:
    temppop.append(i)
for i in range(len(population)):
    fitarr.append(fitness(population[i]))
plt.figure(figsize=(12, 6))
plt.scatter(temppop, fitarr, color = 'red')
plt.title('Kilo Boy Oranı Grafiği3')
plt.xlabel('Kilo')
plt.ylabel('Boy')
plt.xlim(-10,10)
plt.ylim(-3,3)
plt.show()
end_time = time.perf_counter ()
print(end_time- start_time, "seconds")
print(tracemalloc.get_traced_memory())
print(process.memory_percent())
# stopping the library
tracemalloc.stop()