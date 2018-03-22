import tensorflow as tf
import numpy as np

x,y = [],[]
with open("2data.txt") as file:
    for line in file.readlines():
        comb = line.split(" ")
        x.append(int(comb[0]))
        y.append(int(comb[1].replace("\n","")))

val = tf.multiply(tf.add(x,y),tf.add(y,1))

with tf.Session() as sess:
    mul = sess.run(val)
    with open("output.txt","a") as out:
        for x in mul:
            out.write(str(x)+"\n")