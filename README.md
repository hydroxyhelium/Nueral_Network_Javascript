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

### Documentation

More Documentation would be added soon :)
