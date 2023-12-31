module.exports = (outputArray, expectedArray) =>
  outputArray.reduce((sum, val, ind) => sum + Math.pow(val - expectedArray[ind], 2), 0) / outputArray.length;
