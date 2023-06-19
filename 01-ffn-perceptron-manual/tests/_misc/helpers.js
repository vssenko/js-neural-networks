const MultiLayerPerceptron = require('../../src/network/multi-layer-perceptron');

// Example network setup from,
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example

function getExampleNetwork({ silent = false } = {}) {
  const perceptron = new MultiLayerPerceptron({ layerSizes: [2, 2, 2], silent, learningRate: 0.5 });
  //setting the same weights and biases as in example
  perceptron.layers[0][0].outputWires[0].weight = 0.15;
  perceptron.layers[0][0].outputWires[0]._bakedWeight = 0.15;

  perceptron.layers[0][0].outputWires[1].weight = 0.25;
  perceptron.layers[0][0].outputWires[1]._bakedWeight = 0.25;

  perceptron.layers[0][1].outputWires[0].weight = 0.2;
  perceptron.layers[0][1].outputWires[0]._bakedWeight = 0.2;

  perceptron.layers[0][1].outputWires[1].weight = 0.3;
  perceptron.layers[0][1].outputWires[1]._bakedWeight = 0.3;

  perceptron.layers[1][0].bias = 0.35;
  perceptron.layers[1][1].bias = 0.35;

  perceptron.layers[1][0].outputWires[0].weight = 0.4;
  perceptron.layers[1][0].outputWires[0]._bakedWeight = 0.4;

  perceptron.layers[1][0].outputWires[1].weight = 0.5;
  perceptron.layers[1][0].outputWires[1]._bakedWeight = 0.5;

  perceptron.layers[1][1].outputWires[0].weight = 0.45;
  perceptron.layers[1][1].outputWires[0]._bakedWeight = 0.45;

  perceptron.layers[1][1].outputWires[1].weight = 0.55;
  perceptron.layers[1][1].outputWires[1]._bakedWeight = 0.55;

  perceptron.layers[2][0].bias = 0.6;
  perceptron.layers[2][1].bias = 0.6;

  return perceptron;
}

module.exports = {
  getExampleNetwork,
};
