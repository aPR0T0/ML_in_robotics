// Write the base template for the eigen library for a multilayer perceptron
#include <iostream>
#include <Eigen/Dense>


// Define the neural network layer structure
struct NeuralNetworkLayer {
    Eigen::MatrixXd weights; // can be of any dimensions, but it has two dimensions, one depends on the current layer and the other depends on the next layer
    Eigen::VectorXd biases; // can be of any dimensions
};


// Define the neural network structure
struct NeuralNetwork {
    std::vector<NeuralNetworkLayer> layers; // can be of multiple layers
};


// Define the activation function and its derivative
inline Eigen::VectorXd sigmoid(const Eigen::VectorXd& input) {
    return 1.0 / (1.0 + (-input).array().exp());
}

inline Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& input) {
    return input.array() * (1.0 - input.array());
}


// Define the forward propagation function
Eigen::VectorXd forward_propagation(const NeuralNetwork& network, const Eigen::VectorXd& input) {
    Eigen::VectorXd output = input; // Initialized the output vector

    for (const auto& layer : network.layers) {
        output = sigmoid(layer.weights * input + layer.biases); // input is the initial values for the perceptron
    }

    return output;
}


// Define the backpropagation function
void backpropagation(NeuralNetwork& network, const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learning_rate) {

    // Backpropagation
    Eigen::VectorXd output = forward_propagation(network, input);
    Eigen::VectorXd error = target - output;

    // Calculate the gradients
    Eigen::VectorXd gradients = error.array() * sigmoid_derivative(output).array();

    // Update the weights and biases
    for (int i = network.layers.size() - 1; i >= 0; --i) {
        Eigen::MatrixXd weights_gradient = gradients.transpose() * network.layers[i].weights;
        Eigen::VectorXd biases_gradient = gradients.array();

        network.layers[i].weights += learning_rate * weights_gradient;
        network.layers[i].biases += learning_rate * biases_gradient;

        gradients = network.layers[i].weights.transpose() * gradients;
    }

}

// training the network
void train_neural_network(NeuralNetwork& network, const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            backpropagation(network, inputs[i], targets[i], learning_rate);
        }
    }
}

// initializing the network
NeuralNetwork initialize_neural_network(int input_size, std::vector<int> hidden_sizes, int output_size) {
    NeuralNetwork network;

    // Initialize the first layer
    NeuralNetworkLayer layer;
    layer.weights = Eigen::MatrixXd::Random(hidden_size, input_size);
    layer.biases = Eigen::VectorXd::Random(hidden_size[0]);
    network.layers.push_back(layer);

    // initializing the hidden layers
    for (size_t i = 1; i < hidden_sizes.size(); ++i) {
        layer.weights = Eigen::MatrixXd::Random(hidden_sizes[i], hidden_sizes[i - 1]);
        layer.biases = Eigen::VectorXd::Random(hidden_sizes[i]);
        network.layers.push_back(layer);
    }

    // Initialize the output layer
    layer.weights = Eigen::MatrixXd::Random(output_size, hidden_sizes.back());
    layer.biases = Eigen::VectorXd::Random(output_size);
    network.layers.push_back(layer);

    return network;
}

