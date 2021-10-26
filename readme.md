# Distributed Training of Boltzmann Machines

## Hypothesis and Reasoning

A single Boltzmann Machine (BM) can be trained to possess representations for several distinct inputs (i.e. training data).  This experiment tests the hypothesis that multiple BMs can be trained separately, one for each desired input (i.e. class), and then combined into a single BM which then possesses all of the learned representations.  Thus, a single BM with all representations can be trained in a distributed system, i.e. in parallel (e.g. each "agent" learning one representation).


## Experiment and Result

The hypothesis is tested by training 10 BMs separately, each on a single digit of the MNIST dataset, followed by summing their weights to form a single BM, and then verifying that this BM, with no further training, possesses representations for all digits. 

This experiment concludes affirmatively the hypothesis. Code is provided to replicate the experiment. To verify the results, clone this repository, install the dependencies and run ```test.js```.  The Node.js runtime is required.  The program uses the C++ bindings for Tensorflow.js, no GPU support required.  

For each digit, the console will print a training error and a picture of the results. Then the 10 BM weights are summed and used for a single BM. This single BM is queried for all digits, and the console prints a picture of the results. (Results must be verified visually.) 

```
git clone ...
npm i
node test.js -e 8 -s 4 
# -e is number of training cycles per digit
# -s is the size of the weights as multiples of size of a single input (28x28)
```
The BMs created are of the restricted form, and trained using the instructions given in "A Practical Guide to Training Restricted Boltzmann Machines", by Geoffrey Hinton. The code in ```data.js``` and the subset of the MNIST dataset is forked from the tensorflow.js repository.

## Contact

The Author can be reached by emailing ```science``` at ```folkstack.com```. 
