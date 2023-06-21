const tf = require('@tensorflow/tfjs-node');

/**
 *
 * @param {tf.Sequential} network
 * @param {tf.data.Dataset} dataset
 */
async function testNetwork(network, dataset) {
  //NOTE: evalutaion seems not working idk

  console.log('Start testing network.');

  const result = await network.evaluateDataset(dataset);

  console.log('Finished testing network');
  console.log('result: ', JSON.stringify(result));
}

/**
 *
 * @param {tf.Sequential} network
 * @param {Number} label
 * @param {tf.Tensor} inputTensor
 */
async function checkNetworkWorking(network, label, inputTensor) {
  console.log('Checking network can be executed at all');

  const prediction = await network.predict(inputTensor);
  console.log(`Result of prediction label ${label}:`);
  console.log(prediction.arraySync());
}

module.exports = {
  testNetwork,
  checkNetworkWorking,
};
