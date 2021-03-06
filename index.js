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
  console.log('Data', await trainData.toArray());
  console.log('Data', await data.toArray());

  // At the top are the four features, in the middle is a hidden layer with five nodes, 
  // and at the bottom of the three nodes that we'll use for the classification. 

  // 2. Set model
  const model = tf.sequential();

  // Input features = Sigmoid (0-1 prediction)
  // Then we add the hidden layer with five neurons. By specifying the input shape with the number of features, 
  // which is calculated to be four
  model.add(
    tf.layers.dense({
      inputShape: [numOfFeatures],
      activation: "sigmoid",
      units: 5,
    })
  );

  // The softmax function is a more generalized logistic activation function which is used for multiclass classification.
  // Outputs are 3 classes: setosa, virgincia, versicolor
  // Then we add the three neurons at the bottom activating them with a Softmax function to get the probability that the 
  // pattern will match the neuron for that class of flower
  model.add(
    tf.layers.dense({
      activation: "softmax",
      units: 3,
    })
  );

  // Computes the crossentropy loss between the labels and predictions.
  // Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
  // Learning rate
  // Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. Adam combines the best 
  // properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
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
