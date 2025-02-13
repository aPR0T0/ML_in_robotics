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

Eigen::VectorXd forward_propagation(const NeuralNetwork& network, const Eigen::VectorXd& input) {
    Eigen::VectorXd output = input; // Initialized the output vector

    for (const auto& layer : network.layers) {
        std::cout << layer.weights.rows() << " " << layer.weights.cols() << " " << output.rows() << " " << output.cols() <<std::endl;
        output = layer.weights.transpose() * output + layer.biases ; // input is the initial values for the perceptron
        // print out the output
        std::cout << "Output: " << output.transpose() << std::endl;
        // output = output.array().exp(); // apply the activation function (exp)
        // output = output.array() / output.array().sum(); // normalize the output to a probability distribution
    }

    return output;
}

int main(){

    // initialize the neural network
    NeuralNetwork network;

    // initialize the first layer
    NeuralNetworkLayer layer1;
    layer1.weights = Eigen::MatrixXd::Random(4, 3);
    layer1.biases = Eigen::VectorXd::Random(3);
    network.layers.push_back(layer1);

    // initialize the second layer
    NeuralNetworkLayer layer2;
    layer2.weights = Eigen::MatrixXd::Random(3, 1);
    layer2.biases = Eigen::VectorXd::Random(1);
    network.layers.push_back(layer2);

    // initialize the output layer
    // NeuralNetworkLayer layer3;
    // layer3.weights = Eigen::MatrixXd::Random(1, 3);
    // layer3.biases = Eigen::VectorXd::Random(1);
    // network.layers.push_back(layer3);

    // print the weights and biases
    std::cout << "Weights and Biases:" << std::endl;
    for (const auto& layer : network.layers) {
        std::cout << "Weights: " << std::endl << layer.weights << std::endl;
        std::cout << "Biases: " << std::endl << layer.biases << std::endl << std::endl;
    }

    // print the output from the forward propagation
    Eigen::VectorXd input = Eigen::VectorXd::Random(4);
    Eigen::VectorXd output = forward_propagation(network, input);
    std::cout << "Input: " << input.transpose() << ", Output: " << output.transpose() << std::endl;

    return 0;
}