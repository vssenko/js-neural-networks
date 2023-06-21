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
function convertImageToPlainTensor(imageData) {
  //I really don't know why it should be 2d array instead of just 1d [784]

  return new TensorImageData({
    input: tf.tensor2d(_normalizeInputArray(imageData.input), [1, 784]),
    output: tf.tensor2d(imageData.output, [1, 10]),
    label: imageData.label,
  });
}

/**
 * @param {import('./train-data-provider').ImageData} imageData
 */
function convertImageTo2dTensor(imageData) {
  const tensorAs1d = tf.tensor1d(_normalizeInputArray(imageData.input));
  const reshapedTensor = tensorAs1d.reshape([1, 28, 28, 1]);

  return new TensorImageData({
    input: reshapedTensor,
    output: tf.tensor2d(imageData.output, [1, 10]),
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
