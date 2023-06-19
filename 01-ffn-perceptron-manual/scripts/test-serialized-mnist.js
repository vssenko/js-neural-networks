const path = require('path');
const serializer = require('../src/network-management/network-serializer');
const trainer = require('../src/network-management/network-trainer');
const mnistProvider = require('../src/mnist/train-data-provider');

let filePath = process.argv[2];
if (!filePath) throw new Error('No file was provided');

if (filePath.startsWith('./')) {
  filePath = path.join(process.cwd(), filePath);
}

console.log(`Using file "${filePath}"`);

const network = serializer.deserializeFromFile(filePath);

const mnistTestData = mnistProvider.getTestData();

trainer.test(network, mnistTestData);

console.log('Example run on first 10 mnist test images: ');

for (let i = 0; i < 10; i++) {
  const sample = mnistTestData[i];
  network.run(sample.input);

  console.log(`LABEL ${sample.label}  :  ${network.getNetworkPrettifiedOutput()}`);
}
