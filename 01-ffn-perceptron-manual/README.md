# ml-digits-recognition
Main goal of this application is to show how neural network works step-by-step.

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

# Code

Network itself and trainer are separated things.

About network.

Main code idea is to reprecent neurons and wires as objects, with all the parameters set inside them.  
Neuron has `.inputWires` and `.outputWires`.
Wire has `.inputNeuron` and `.outputNeuron`.
Remember that object variable in JS is a reference to object, so  
`neuron === neuron.outputWires[0].inputNeuron`  
as well as  
`neuron.outputWires[0] === neuron.outputWires[0].outputNeuron.inputWires[0]`.

This gives you more natural understanding how actually numbers are flowing in NN. From my perspective it is easier to understand than analyzing 2d-arrays of numbers.

About trainer - it is simple.

# What could be better

Oh, a lot of things.

First, it is slow. No GPU, no even fast matrix multiplications. Bit by bit, byte by byte :-). With the fact that it is just for educational purpose, it should be fine.

Second, no batch backpropagation. It does backpropagation after each unsuccessfull execution, instead of aggregating deviations and performing backpropagation once in N turns (or after some treshold). Once again, the simpler NN is, the better.

Third, here you find dynamic learning rate implemented, which is better than static, but it is just fading over time. For better results it could be based on how bad error is.

Forth, for loss calculation mean square method is used. It is considered definitely not the best choice with binary classification.

Fifth, project is written in pure JS, no Typescript. After getting back to it, I found that it would be easier to explore soft Typescript project than that. Next time probably will use it.

And finally, I couldn't say that it has fantastic results :-) Though I didn't have a goal to score max score.