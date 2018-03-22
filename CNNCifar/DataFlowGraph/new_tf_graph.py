from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Dataflow is a common programming model for parallel computing. In a dataflow graph, the nodes represent units of computation, and the edges
# represent the data consumed or produced by a computation. For example, in a TensorFlow graph, the tf.matmul operation would correspond to a
# single node with two incoming edges (the matrices to be multiplied) and one outgoing edge (the result of the multiplication).


# basic idea is to outline a tensorflow computation graph with input as list of 2d points (input edge also tf.Tensor object) and then apply a
# function [a+b][b+1] (which acts as node also called tf.Operation) and then an outgoing edge representing 1 D list of computation done.

x = []          # list to store the input 2 dimensional points
y = []          # list to store the input 2 dimensional points

def calculate():
    with open("2data.txt") as file:
        for line in file.readlines():
            comb = line.split(" ")
            x.append(int(comb[0]))
            y.append(int(comb[1].replace("\n","")))

calculate()

# use default graph but one can explicitly define a graph with some operations like
# tensorflow computational graph it just shows flow and no computation and memory
# graph = tf.Graph()

a = tf.constant(0)
b = tf.constant(0)

add = tf.add(a,b)
multiply = tf.multiply(a,b)

with tf.Session() as sess:
    print("Generating o/p .....")
    with open("output.txt",'a') as out:
        for i in range(0,len(x)):
            res1 = sess.run(add,feed_dict={ a : x[i] , b : y[i]})
            res2 = sess.run(add,feed_dict={ a : y[i] , b : 1})
            res3 = sess.run(multiply,feed_dict={ a : res1 , b : res2})
            out.write(str(res3)+"\n")




