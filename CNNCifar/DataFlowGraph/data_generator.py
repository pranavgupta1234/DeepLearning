import numpy as np

with open("2data.txt","a") as file:
    for i in range(0,1000):
        file.write(str(np.random.randint(1,1000))+" "+str(np.random.randint(1,1000))+"\n")