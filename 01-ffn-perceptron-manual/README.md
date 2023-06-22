# 01-ffn-perceptron-manual
Main goal of this console application is to show how neural network works step-by-step.

You may be interested if you:
- want to learn NN, Perceprton, Feed forwarding and Backpropagation.
- want to see how NN works step-by-step with no Matrix multiplication.
- know JavaScript and want to stick with that.

Current implementation is based on plain Multi Layer Perceptron with on-line training (no batches, backpropagation is done after each sample run).

# Presteps
1. npm i
2. npm run download-dataset

# Scripts
Things you can do with that:

## `npm test`
Run tests to be ensured that everything works correctly and see neural network each-step logs for better understanding

## `npm run test-xor`
Script to run NN "Hello World" - XOR operation prediction.

## `npm run train-mnist`
Script to run NN over MNIST training set, and after that test it over MNIST test set.
Optionally, you can provide one parameter, file path to already serialized NN if you want to use already existing NN instead of creating new one. 

## `npm run test-serialized-mnist`
One parameter is required, file path to serialized NN. Should be something like `npm run test-serialized-mnist ./serialized-network-samples/my-sample.json`
Script will load that NN and test it against test MNIST data.

# Implementation

Here is brief description how application was written.

First, I have created neural network as class, which can build layers of neurons.
Each neuron has `.inputWires` and `.outputWires`.
Wire has `.inputNeuron` and `.outputNeuron`.
Remember that object variable in JS is a reference to object, so  
`neuron === neuron.outputWires[0].inputNeuron`  
as well as  
`neuron.outputWires[0] === neuron.outputWires[0].outputNeuron.inputWires[0]`.

This gives you more natural understanding how actually numbers are flowing in NN. From my perspective it is easier to understand than analyzing 2d-arrays of numbers.

Next, I've created basic math functions and methods for calculating neural network output based on input (`run` method in NN and `feedForward` method in Neuron).

After that, the most funny part is to implement very explicit backpropagation, which can show you the logic on each neuron. (Check `backpropagateError` method in NN and `backpropagateForOutputLayer`/`backpropagateForHiddenLayer` in Neuron).

After these steps and finding all the bugs, it's time to build things around NN.

In `/scripts/download-dataset` you will find code to download MNIST digits files.

In `/src/mnist` there is code for parsing that files.

In `/src/network-management` exists our trainer for neural network, which get the data, run for each sample, and backpropagate errors. No batches were implemented for simplicity of understanding basic-basic NN workflow. Also in this folder exists serializer, but i think it's pretty trivial.

All entry scripts in application are in `/scripts` folder.