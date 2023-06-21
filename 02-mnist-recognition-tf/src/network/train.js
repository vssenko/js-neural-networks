const tf = require('@tensorflow/tfjs-node');

/**
 * @param {tf.Sequential} network
 * @param {tf.data.Dataset} dataset
 */
async function trainNetwork(network, dataset, { epochs = 1 } = {}) {
  console.log('Start training network...');
  await network.fitDataset(dataset, {
    epochs,
    callbacks: {
      onEpochEnd(epoch, logs) {
        console.log(`Epoch #${epoch} ended. Loss:${logs.loss}, acc:${logs.acc}`);
      },
    },
  });

  console.log('Network is trained');
}

module.exports = {
  trainNetwork,
};
