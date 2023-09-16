## NueralJS - JavaScript Machine Learning Library

NueralJS is a lightweight and flexible JavaScript library for building and training neural networks. With NueralJS, you can easily create feed-forward neural networks and convolutional layers for various machine learning tasks.

### Features

- Create and configure neural network models.
- Define custom architectures with ease.
- Train models using backpropagation.
- Support for feed-forward and convolutional layers.
- Save and load trained models.

### Getting Started

`npm install nueraljs`

### Usage

```javascript
const { NeuralNetwork, Layer, ActivationFunction } = require("nueraljs");
``;

// Create a neural network
const nn = new NeuralNetwork();

// Add layers to the network
nn.addLayer(new Layer(2, 3, ActivationFunction.SIGMOID));
nn.addLayer(new Layer(3, 1, ActivationFunction.SIGMOID));

// Compile the network
nn.compile();
```

### Training a Model

```javascript
const trainingData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const options = {
  epochs: 10000,
  learningRate: 0.1,
};

nn.train(trainingData, options);
```

### Making Predictions

```javascript
const input = [0, 1];
const prediction = nn.predict(input);
console.log(prediction); // [0.987]
```

### Documentation

More Documentation would be added soon :)
