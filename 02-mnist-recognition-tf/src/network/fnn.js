const tf = require('@tensorflow/tfjs-node');

function buildNetwork() {
  console.log('Creating FNN network.');
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

module.exports = {
  buildNetwork,
};
