import numpy as np
import random as rnd
import math as mt
from matplotlib import pyplot as plt
iterasyon=10
crosover_rate=0.25
pop_size=10
gen_size=4

def create_chromosome():
    return [rnd.randint(0,1) for x in range(0,gen_size)]
def create_initial_population():
    return [create_chromosome() for x in range(0,pop_size)]
def fitness(cr):
    return 1/(1+abs(fitness_function(cr)))
def fitness_function(x):
    binary_to_decimal = int("".join(str(x) for x in x), 2)
    return mt.sin(binary_to_decimal)+mt.sin(10*binary_to_decimal/ 3)
def probability(fitness_values):
    P=[]
    total=sum(fitness_values)
    for f in fitness_values:
        P.append(f/total)
    print(P)
    return P
def crossover(p1, p2):
    o1 = []
    o2 = []
    c = rnd.randint(1, gen_size - 1)
    print("Cut point:", c)
    o1[:c] = p2[:c]
    o1[c:] = p1[c:]
    o2[:c] = p1[:c]
    o2[c:] = p2[c:]

    return o1, o2
def mutasyon(mut):
    temp=[]
    temp=mut[:]
    gen=rnd.randint(0,1)
    index=rnd.randint(0,gen_size-1)
    temp[index]=gen
    return temp

population=create_initial_population()
fitness_values=[]
for c in range(len(population)):
    binary_to_decimal = int("".join(str(x) for x in population[c]), 2)
    if binary_to_decimal > 10:
        population[c]= [1, 0, 1, 0]
    fitness_values.append(fitness(population[c]))
epok=0


while epok<iterasyon:
    P=probability(fitness_values)
    C=np.cumsum(P)
    rulet_parents=[]
    print("cumsum ", C)
    for i in range(0,len(C)):
        r=rnd.random()
        for j in range(0,len(C)):
            if C[j]>r:
                rulet_parents.append(j)
                break


    for c, f,p in zip(population,fitness_values,P):
        print(c, " ",f," ",p)
    print(C)

    print(rulet_parents)

    crosover_parents=[]
    k=0
    while k<pop_size:
        r=rnd.random()
        if(r<crosover_rate):
            if(rulet_parents[k] not in crosover_parents):
                crosover_parents.append(rulet_parents[k])
        k=k+1
    print("Caprazalnacak bireyler:",crosover_parents)

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
    print("crossover sonrasi populasyon")
    for c, f in zip(population,fitness_values):
        print(c, " ",f )
    for r in range(0,5):
        mut=mutasyon(population[rnd.randint(0,len(population)-1)])
        population.append(mut)
        fitness_values.append(fitness(mut))

    print("mutasyon sonrasi populasyon")
    for c, f in zip(population,fitness_values):
        print(c, " ",f )

    zip_list=zip(fitness_values,population)

    sort_list=sorted(zip_list)

    for f,p in list(sort_list):
        print(f," ",p)

    p=len(population)

    while(p>pop_size):
        sort_list.pop()
        p=p-1
    print("elitizm sonrasi")

    for f,p in list(sort_list):
        print(f," ",p)

    population=[]
    fitness_values=[]

    for f,p in list(sort_list):
        population.append(p)
        fitness_values.append(f)
    epok+=1

print("Son populasyon")
for c, f in zip(population,fitness_values):
        print(c, " ",f )
print("En iyi birey:",population[0]," fitness:",fitness_values[0])
test = [1,0,0,0]
binary_to_decimal = int("".join(str(x) for x in test), 2)
temppop=[]
fitarr=[]
for i in population:
    binary_to_decimal = int("".join(str(x) for x in i), 2)
    temppop.append(binary_to_decimal)
for i in range(len(population)):
    fitarr.append(fitness_function(population[i]))
plt.figure(figsize=(12, 6))
plt.scatter(temppop, fitarr, color = 'red')
plt.title('Kilo Boy Oranı Grafiği3')
plt.xlabel('Kilo')
plt.ylabel('Boy')
plt.show()
test = [-1,1,0,0]
binary_to_decimal = int("".join(str(x) for x in test),2)
print(binary_to_decimal)