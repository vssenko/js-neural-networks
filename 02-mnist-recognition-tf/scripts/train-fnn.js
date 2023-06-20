const config = require('../src/config');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const mnistDataProvider = require('../src/mnist/train-data-provider');

async function main() {
  console.log('Starting exuction FNN flow.');
  const network = createNetwork();

  let trainData = await mnistDataProvider.getTrainData();

  await checkNetworkWorking(network, trainData[0]);

  wait(500);

  await trainNetwork(network, trainData);

  wait(1000);

  await checkNetworkWorking(network, trainData[0]);

  wait(1000);

  trainData = null;
  const testData = await mnistDataProvider.getTestData();
  await testNetwork(network, testData);
}

function createNetwork() {
  console.log('Creating TF network.');
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [28 * 28],
      units: 128,
      activation: 'relu',
    })
  );

  model.add(
    tf.layers.dense({
      units: 10,
      activation: 'softmax',
    })
  );

  console.log('Compiling TF network.');
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: 'adam',
    metrics: ['accuracy'],
  });

  console.log('Network is compiled.');
  console.log(model.summary());

  return model;
}

/**
 * @param {tf.Sequential} network
 * @param {mnistDataProvider.ImageData[]} data
 */
async function trainNetwork(network, data) {
  console.log('Preparing data to TF.');
  data = data.map(tensorizeImageData);

  const dataset = tf.data
    .zip({
      xs: tf.data.array(data.map((d) => d.input)),
      ys: tf.data.array(data.map((d) => d.output)),
    })
    .shuffle(4);

  console.log('Start training network...');
  const result = await network.fitDataset(dataset, {
    epochs: 1,
    callbacks: {
      onEpochEnd(epoch, logs) {
        console.log(`Epoch #${epoch} ended. Loss:${logs.loss}, acc:${logs.acc}`);
      },
    },
  });

  console.log('Network is trained');
}

/**
 * @param {tf.Sequential} network
 * @param {mnistDataProvider.ImageData[]} data
 */
async function testNetwork(network, data) {
  console.log('Start testing network.');

  console.log('Preparing data.');
  data = data.map(tensorizeImageData);

  const dataset = tf.data.zip({
    xs: tf.data.array(data.map((d) => d.input)),
    ys: tf.data.array(data.map((d) => d.output)),
  });

  await network.evaluateDataset(dataset, {
    verbose: true,
  });

  console.log('Finished testing network');
}

/**
 *
 * @param {tf.Sequential} network
 * @param {mnistDataProvider.ImageData} imageData
 */
async function checkNetworkWorking(network, imageData) {
  console.log('Checking network can be executed at all');
  const tensorizedImageData = tensorizeImageData(imageData);

  const prediction = await network.predict(tensorizedImageData.input);
  console.log(`Result of prediction label ${tensorizedImageData.label}:`);
  console.log('actual: ', await prediction.data());
  console.log('expected: ', imageData.output);
}

/**
 * @param {mnistDataProvider.ImageData} imageData
 */
function tensorizeImageData(imageData) {
  return {
    input: tf.tensor2d(
      imageData.input.map((i) => i / 255),
      [1, 784]
    ),
    output: tf.tensor2d(imageData.output, [1, 10]),
    label: imageData.label,
  };
}

function wait(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

Promise.resolve()
  .then(() => main())
  .catch((e) => {
    console.error('Error!');
    console.error(e);
    process.exit(-1);
  });
