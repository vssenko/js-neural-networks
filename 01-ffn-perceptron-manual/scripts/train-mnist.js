const path = require('path');
const { MultiLayerPerceptron } = require('../src/network');
const serializer = require('../src/network-management/network-serializer');
const trainer = require('../src/network-management/network-trainer');
const mnistProvider = require('../src/mnist/train-data-provider');

const mnistTrainData = mnistProvider.getTrainData();

let network = null;

let filePath = process.argv[2];
if (filePath) {
  filePath = path.join(process.cwd(), filePath);
  console.log(`Using file "${filePath}"`);
  network = serializer.deserializeFromFile(filePath);
} else {
  network = new MultiLayerPerceptron({
    layerSizes: [784, 256, 64, 10],
  });
}

network.learningRate = 0.03;
network.minLearningRate = 0.001;
network.learningRateDecayStep = 15000;

trainer.train(network, mnistTrainData, {
  silent: false,
  epochesCount: 5,
  errorTreshold: 0.005,
  serializeAfterEpoch: true,
});

const mnistTestData = mnistProvider.getTestData();

trainer.test(network, mnistTestData);

console.log('Example run on first 10 mnist test images: ');

for (let i = 0; i < 10; i++) {
  const sample = mnistTestData[i];
  network.run(sample.input);

  console.log(`LABEL ${sample.label}  :  ${network.getNetworkPrettifiedOutput()}`);
}
