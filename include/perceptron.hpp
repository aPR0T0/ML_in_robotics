#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <iostream>
#include <vector>
#include <optional>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <algorithm>
#include <eigen3/Eigen/Dense>


// Define the neural network layer structure
struct NeuralNetworkLayer {
    Eigen::MatrixXd weights; // can be of any dimensions, but it has two dimensions, one depends on the current layer and the other depends on the next layer
    Eigen::VectorXd biases; // can be of any dimensions
};

inline int number_of_classes;

// Define the neural network structure
struct NeuralNetwork {
    std::vector<NeuralNetworkLayer> layers; // can be of multiple layers
};


// Define the activation function and its derivative
inline Eigen::VectorXd sigmoid(const Eigen::VectorXd& input) ;
inline Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& input) ;

// Define the forward propagation function
std::vector<Eigen::VectorXd> forward_propagation(const NeuralNetwork& network, const Eigen::VectorXd& input);

// Define the backpropagation function
float backpropagation(NeuralNetwork& network, const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learning_rate);
// training the network
void train_neural_network(NeuralNetwork& network, const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs, double learning_rate, size_t num_samples);

// initializing the network
NeuralNetwork initialize_neural_network(int input_size, std::vector<int> hidden_sizes, int output_size) ;

#endif