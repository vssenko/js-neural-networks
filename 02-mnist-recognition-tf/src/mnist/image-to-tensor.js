const tf = require('@tensorflow/tfjs-node');

class TensorImageData {
  constructor({ input, output, label }) {
    this.input = input;
    this.output = output;
    this.label = label;
  }
}

/**
 * @param {import('./train-data-provider').ImageData} imageData
 */
function convertImageToPlainTensor(imageData, { asBatch = true } = {}) {
  const inputShape = asBatch ? [1, 784] : [784];
  const outputShape = asBatch ? [1, 10] : [10];
  return new TensorImageData({
    input: tf.tensor(_normalizeInputArray(imageData.input), inputShape),
    output: tf.tensor(imageData.output, outputShape),
    label: imageData.label,
  });
}

/**
 * @param {import('./train-data-provider').ImageData} imageData
 */
function convertImageTo2dTensor(imageData, { asBatch = true } = {}) {
  const inputShape = asBatch ? [1, 28, 28, 1] : [28, 28, 1];
  const outputShape = asBatch ? [1, 10] : [10];

  return new TensorImageData({
    input: tf.tensor(_normalizeInputArray(imageData.input), inputShape),
    output: tf.tensor(imageData.output, outputShape),
    label: imageData.label,
  });
}

/**
 *
 * @param {Number[]} input
 */
function _normalizeInputArray(input) {
  return input.map((i) => i / 255);
}

module.exports = {
  convertImageToPlainTensor,
  convertImageTo2dTensor,
};
