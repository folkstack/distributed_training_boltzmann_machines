# Distributed Training of Boltzmann Machines

## Hypothesis and Reasoning

A single Boltzmann Machine (BM), like most neural networks, can be trained to possess representaions for several inputs, i.e. images of digits 0-9.  This experiment tests the hypothesis that multiple BMs can be trained separately, one for each desired input, and then combined into a single BM which then possesses all of the learned representations.  Thus, a single BM could be trained in a distributed system.

BMs are a energy based neural networks, rather than probabilistic. The goal of training is to create a representation of the learned data while lowering the energy in the network. A more common energy representation are samples of audio in the amplitude domain. Multiple audio streams can be added together into a single channel.  This clue was the reasoning for the hypothesis.  

## Experiment and Result

The hypothesis is tested by training 10 BMs separately, each on a single digit of the MNIST dataset, followed by summing their weights to form a single BM, and then verifying that this BM, with no further training, possesses representations for all digits. 

This experiment concludes affirmatively the hypothesis. Code is provided to replicate the experiment. To verify the results, clone this repository, install the dependencies and run ```test.js```.  The Node.js runtime is required.  The program uses the C++ bindings for Tensorflow.js, no GPU support required.  

For each digit, the console will print a training error and a picture of the results. Then the 10 BM weights are summed and used for a single BM. This single BM is queried for all digits, and the console prints a picture of the results. (Results must be verified visually.) 

Notably, the results are better after the weights are summed to form a single BM.  The likely reason is that the single digit weights, which are the same size as the summed weight, contain extra noise, b 
```
git clone ...
npm i
node test.js -e 8 -s 4 
# -e is number of training cycles per digit
# -s is the size of the weights as multiples of size of a single input (28x28)
```
The BMs created are of the restricted form, and trained using the instructions given in "A Practical Guide to Training Restricted Boltzmann Machines", by Geoffrey Hinton. The code in ```data.js``` and the subset of the MNIST dataset is forked from the tensorflow.js repository.

