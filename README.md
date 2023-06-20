# js-neural-networks
A bunch of neural networks examples in Javascript.

# Introduction
This github repository will contain a bunch of subprojects representing implementation of different NN architectures for different tasks. Some NN will be created fully manually, some of them using TensorFlow.

Each project is completely independent from others, some code duplication may occur, but this is done for decreasing overall complexity as much as possible.

*WORK IN PROGRESS.* Stay tuned.

# Why Javascript?

Just because I like C-like languages more, and Javascript past decade became suitable tool for everything, starting from front end and ending in embedded devices. If you are passionate about neural network and want to understand them better, with examples in your favorite language - you are welcome!

P.S: Probably Typescript will come in next projects.

# Projects

## 01. Feedforward NN. Perceptron.
Manually implemented basic neural network, as variable sized multi layer Perceptron, with trainer, with serialization.
Task: Parse MNIST digits.
[Click here](./01-ffn-perceptron-manual/README.md) to learn more.

## 02. Tensorflow-based digits recognition.
Here we will do the same task as in first project, but using Tensorflow. Two different architectures will be implemented, classic FNN (as in first project) and better-suited for image recognition CNN.
