const relu = require('../mathFunctions/relu');

const dRelu = require('../mathFunctions/dRelu');

module.exports = {
  name: 'leakyRelu',
  func: relu,
  dFunc: dRelu,
};
