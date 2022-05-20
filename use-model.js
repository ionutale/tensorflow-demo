const tf = require('@tensorflow/tfjs-node');

async function run() {

  const modelUrl = `file://${__dirname}/models/model.json`;
  const model = await tf.loadLayersModel(modelUrl);
  model.summary();

  // model.predict(tf.ones([92, 12]), {batchSize: 10, verbose: true}).print();
  model.predict(
    tf.tensor2d([
      60
    ], [1, 1])
  ).print();
}

run().then(() => {
  console.log('Done')
})