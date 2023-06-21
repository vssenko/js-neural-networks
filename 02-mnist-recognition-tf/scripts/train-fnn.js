const tf = require('@tensorflow/tfjs-node');
const mnistDataProvider = require('../src/mnist/train-data-provider');
const mnistTransformer = require('../src/mnist/image-to-tensor');
const fnnBuilder = require('../src/network/fnn');
const trainer = require('../src/network/train');
const tester = require('../src/network/test');

async function main() {
  console.log('Starting exuction FNN flow.');

  const network = fnnBuilder.buildNetwork();

  let trainData = await mnistDataProvider.getTrainData();

  const firstTrainDataTensorized = mnistTransformer.convertImageToPlainTensor(trainData[0]);
  const firstLabel = trainData[0].label;
  await tester.checkNetworkWorking(network, firstLabel, firstTrainDataTensorized.input);

  let trainDataset = _buildDataset(trainData);
  await trainer.trainNetwork(network, trainDataset);

  await tester.checkNetworkWorking(network, firstLabel, firstTrainDataTensorized.input);

  trainData = trainDataset = null;

  console.log('');
  console.log('');

  const testData = await mnistDataProvider.getTestData();
  const testDataset = _buildDataset(testData);
  await tester.testNetwork(network, testDataset);
}

/**
 *
 * @param {mnistDataProvider.ImageData[]} imageData
 */
function _buildDataset(imageDataArray) {
  const imagesToTensorImages = imageDataArray.map((id) => mnistTransformer.convertImageToPlainTensor(id));

  return tf.data
    .zip({
      xs: tf.data.array(imagesToTensorImages.map((d) => d.input)),
      ys: tf.data.array(imagesToTensorImages.map((d) => d.output)),
    })
    .batch(32)
    .shuffle(4);
}

Promise.resolve()
  .then(() => main())
  .catch((e) => {
    console.error('Error!');
    console.error(e);
    process.exit(-1);
  });
