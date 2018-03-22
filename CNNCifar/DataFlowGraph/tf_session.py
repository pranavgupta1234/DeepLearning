from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

x,y = [],[]
with open("2data.txt") as file:
    for line in file.readlines():
        comb = line.split(" ")
        x.append(int(comb[0]))
        y.append(int(comb[1].replace("\n","")))

# Here val is a Tensor one can check by print(val) but remember tf.multiply() is a tf.Operation(a node) and its returns as tensor
# A tf.Tensor object represents a partially defined computation that will eventually produce a value. TensorFlow
# programs work by first building a graph of tf.Tensor objects, detailing how each tensor is computed based on the
# other available tensors and then by running parts of this graph to achieve the desired results.
val = tf.multiply(tf.add(x,y),tf.add(y,1))

# Here one should also remember that the ops and tensors all are added to default graph
# but one can define his own graph with ops and tensors and execute them partially

with tf.Session() as sess:
    mul = sess.run(val)                             #here it actually computes and returns result
    with open("output.txt","a") as out:
        for x in mul:
            out.write(str(x)+"\n")