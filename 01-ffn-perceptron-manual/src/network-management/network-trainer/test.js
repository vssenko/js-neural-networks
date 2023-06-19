const _ = require('lodash');
const config = require('../../config');

const squaredErrorCostCostFunction = require('../../network/mathFunctions/squaredErrorCost');

function test(network, testData, { errorTreshold } = {}) {
  errorTreshold = errorTreshold || config.training.defaultErrorTreshold;

  console.log(`Starting testing network, error treshhold is ${errorTreshold}`);

  let totalProceed = 0;
  let correctSquareCostAnswerCount = 0;
  let correctExtremumAnswerCount = 0;
  for (let sample of testData) {
    const data = sample.input;
    const expected = sample.output;
    const resultArray = network.run(data);

    const isCorrectBySquareCost = checkIfCorrectBySquareCost({
      actual: resultArray,
      expected,
      errorTreshold,
    });
    if (isCorrectBySquareCost) {
      correctSquareCostAnswerCount++;
    }

    const isCorrectByExtremum = checkIfCorrectByCompletelyWrongExtremumCheck({
      actual: resultArray,
      expected,
    });
    if (isCorrectByExtremum) {
      correctExtremumAnswerCount++;
    }
    totalProceed++;
  }

  console.log('Final test results:');
  console.log(`Total proceed: ${totalProceed}`);
  console.log(`Correct by square cost answers: ${correctSquareCostAnswerCount}`);
  console.log(`Percent of correct answers: ${(correctSquareCostAnswerCount / totalProceed) * 100}`);
  console.log(`Correct by extremum answers: ${correctExtremumAnswerCount}`);
  console.log(`Percent of correct by extremum answers: ${(correctExtremumAnswerCount / totalProceed) * 100}`);
}

module.exports = test;

function checkIfCorrectBySquareCost({ actual, expected, errorTreshold }) {
  const error = squaredErrorCostCostFunction(actual, expected);

  return error <= errorTreshold;
}

function checkIfCorrectByCompletelyWrongExtremumCheck({ actual, expected }) {
  const maxExpectedIndex = _.indexOf(expected, _.max(expected));
  const maxActualIndex = _.indexOf(actual, _.max(actual));

  return maxExpectedIndex === maxActualIndex;
}
