import * as tf from "@tensorflow/tfjs-node-gpu";

const CLASS_NAMES = ["Setosa", "Virginica", "Versicolor"];
const AXIS = 1;
const CSV_URL = `file://${__dirname}/iris.csv`;
const TEST_VAL = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);

async function start() {
  // 1. Set data
  const trainData = tf.data.csv(CSV_URL, {
    columnConfigs: {
      species: {
        isLabel: true,
      },
    },
  });

  const numOfFeatures = (await trainData.columnNames()).length - 1;

  const data = trainData
    .map(({ xs, ys }) => {
      const labels = [
        ys.species == "setosa" ? 1 : 0,
        ys.species == "virgincia" ? 1 : 0,
        ys.species == "versicolor" ? 1 : 0,
      ];
      return { xs: Object.values(xs), ys: Object.values(labels) };
    })
    .batch(10);

  // 2. Set model
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [numOfFeatures],
      activation: "sigmoid",
      units: 5,
    })
  );

  model.add(
    tf.layers.dense({
      activation: "softmax",
      units: 3,
    })
  );

  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.06),
  });

  model.summary();

  // 3. Train model
  await model.fitDataset(data, {
    epochs: 100,
    callbacks: {
      onBatchEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch} Loss: ${logs.loss}`);
      },
    },
  });

  // 4. Make prediction
  const prediction = model.predict(TEST_VAL);
  const pIndex = tf.argMax(prediction, AXIS).dataSync();
  console.log(CLASS_NAMES[pIndex]);
}

start();
