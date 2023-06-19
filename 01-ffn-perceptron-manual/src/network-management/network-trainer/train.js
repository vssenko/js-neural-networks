const _ = require('lodash');
const config = require('../../config');
const networkSerializer = require('../network-serializer');
const squaredErrorCostCostFunction = require('../../network/mathFunctions/squaredErrorCost');

const batchCheckIteration = 10000;

function train(network, trainData, { epochesCount, errorTreshold, silent = true, serializeAfterEpoch = false, shuffle = true } = {}) {
  errorTreshold = errorTreshold || config.training.defaultErrorTreshold;
  epochesCount = epochesCount || config.training.defaultEpochcesCount;

  console.log('Training network...');
  console.log(`Satisfying cost threshold is ${errorTreshold}`);

  let costsSum;
  let batchIndex;
  let currentCost = 10;
  let stopTraining = false;

  for (let epoch = 1; epoch <= epochesCount; epoch++) {
    costsSum = 0;
    batchIndex = 0;
    if (stopTraining) {
      break;
    }

    const shuffledTrainData = shuffle ? _.shuffle(trainData) : trainData;

    for (let i = 0; i < shuffledTrainData.length; i++) {
      const sample = shuffledTrainData[i];
      const data = sample.input;
      const expected = sample.output;
      const result = network.run(data);

      currentCost = squaredErrorCostCostFunction(result, expected);

      costsSum += currentCost;
      batchIndex++;
      if (currentCost > errorTreshold) {
        network.backpropagateError(expected);
      }

      if (batchIndex >= batchCheckIteration) {
        const medianCost = costsSum / batchCheckIteration;
        if (!silent) {
          console.log(`Epoch ${epoch}. Processing data sample #${i}. Current median cost : ${costsSum / batchCheckIteration}`);
          console.log(`Network: learning rate = ${network.learningRate}`);
          console.log(`Label ${sample.label}. Network output: ${network.getNetworkPrettifiedOutput()}`);
          console.log(`Current error cost = ${currentCost}`);
        }

        costsSum = 0;
        batchIndex = 0;

        if (medianCost <= errorTreshold) {
          console.log(`Cost is satisfying (${medianCost}), stop training at Epoch ${epoch}, iteration ${i}.`);
          stopTraining = true;
          break;
        }
      }
    }

    if (serializeAfterEpoch) {
      networkSerializer.serializeAndSave(network);
    }
  }

  if (!stopTraining) {
    console.log('Warning: training was not successfull');
  }
}

module.exports = train;
