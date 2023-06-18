const {MultiLayerPerceptron} = require('../src/network');
const trainer = require('../src/network-management/network-trainer');
const mnistProvider = require('../src/mnist/train-data-provider');

const mnistTrainData = mnistProvider.getTrainData();

const network = new MultiLayerPerceptron({
  layerSizes: [784, 256, 128, 64, 10],
  learningRate: 0.015,
  learningRateDecayStep: 20000
});

trainer.train(
  network,
  mnistTrainData,
  { silent: false, epochesCount: 5, successfullStreak: 50, errorTreshold: 0.001, serializeAfterEpoch: true });

const mnistTestData = mnistProvider.getTestData();

trainer.test(network, mnistTestData);

console.log('Example run on first 10 mnist test images: ');

for (let i =0; i < 10; i++){
  const sample = mnistTestData[i];
  const result = network.run(sample.input);

  console.log(`${sample.label}:${result}`);
}