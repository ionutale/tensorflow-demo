// import * as tf from '@tensorflow/tfjs';
const  tf = require('@tensorflow/tfjs-node');

const csvUrl = 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';
const peopleAgeRange = `file://${__dirname}/people-age-range.csv`;

console.log('start', peopleAgeRange);

async function run() {
  // We want to predict the column "medv", which represents a median value of a
  // home (in $1000s), so we mark it as a label.
  const csvDataset = tf.data.csv(
    peopleAgeRange, {
      columnConfigs: {
        Nom: {
          isLabel: true,
        }
      }
    });
  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;
// console.log(csvDataset)
// console.log(numOfFeatures)
  // Prepare the Dataset for training.
  const flattenedDataset =
    csvDataset
    .map(({xs, ys}) => {
      // Convert xs(features) and ys(labels) from object form (keyed by column
      // name) to array form.
      // console.log({xs: Object.values(xs), ys: Object.values(ys)})
      return {xs: Object.values(xs), ys: Object.values(ys)};
    })
    .batch(10);

  // Define the model.
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1
  }));
  model.compile({
    optimizer: tf.train.sgd(0.000001),
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  // Fit the model using the prepared Dataset
  await model.fitDataset(flattenedDataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch, logs.loss);
      }
    }
  });

  
  model.summary();

  const saveResult = await model.save(`file://${__dirname}/models`);
  // This will trigger downloading of two files:
  //   'mymodel.json' and 'mymodel.weights.bin'.
  console.log(saveResult);


}

run().then( () => {
  console.log('Done')
})
