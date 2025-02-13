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
        // print the dimensions of the layer weights and the output vector
        std::cout << layer.weights.rows() << " " << layer.weights.cols() << " " << output.rows() << " " << output.cols() << std::endl;
        // output = sigmoid(layer.weights.transpose() * output + layer.biases); // apply the sigmoid activation function
        output = sigmoid(layer.weights * output + layer.biases); // input is the initial values for the perceptron
        // print out the output
        std::cout << "Output: " << output.transpose() << std::endl;
    }

    return output;
}


// Define the backpropagation function
float backpropagation(NeuralNetwork& network, const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learning_rate) {

    // Backpropagation
    Eigen::VectorXd output = forward_propagation(network, input);
    Eigen::VectorXd error = target - output;
    // print out the error
    std::cout << "Error: " << error.transpose() << std::endl;
    // Calculate the gradients
    Eigen::VectorXd gradients = error.array() * sigmoid_derivative(output).array();

    // print out the gradients
    std::cout << "Gradients: " << gradients.transpose() << std::endl;

    // Adjust the weights and biases using gradient descent
    // Note: This is a simplified version of gradient descent. For a more efficient and accurate implementation, consider using a more advanced optimization algorithm like Adam or RMSprop.
    // Update the weights and biases
    for (int i = network.layers.size() - 1; i >= 0; i--) {
        // print shape of gradients and weights
        std::cout << "Weights before update: " << std::endl << network.layers[i].weights.rows() << " "  << network.layers[i].weights.cols() <<std::endl;
        // std::cout << "Biases before update: " << std::endl << network.layers[i].biases << std::endl << std::endl;

        std::cout << "Gradients before update: " << gradients.transpose().rows() << " " << gradients.transpose().cols()<< std::endl;

        // Calculate the gradients for the current layer
        // Note: This is a simplified version of backpropagation. For a more efficient and accurate implementation, consider using a more advanced optimization algorithm like Adam or RMSprop
        Eigen::MatrixXd weights_gradient;
        Eigen::VectorXd biases_gradient = gradients.transpose();
        std::cout << "Weights.col(i): " << std::endl << network.layers[i].weights<< std::endl;
        
        for (int j = network.layers[i].weights.cols() - 1; j >= 0; j--){
            // print dimensions of weigts.col(i) and dimensions of gradients
            std::cout<<gradients<<std::endl;
            weights_gradient = gradients * network.layers[i].weights.col(j);
            // print weights_gradient
            
            std::cout << "Weights gradient: " << weights_gradient << std::endl;
            network.layers[i].weights.col(j) += learning_rate * weights_gradient.transpose();
        }
        // print weights_gradient
        std::cout << "Weights gradient: " << weights_gradient.rows() << " " << weights_gradient.cols() << std::endl;
        std::cout << "Biases gradient: " << biases_gradient.rows() << " " << biases_gradient.cols() << std::endl;

        // Update the weights and biases
        // Note: This is a simplified version of gradient descent. For a more efficient and accurate implementation, consider using a more advanced optimization algorithm like Adam or RMSprop.
        network.layers[i].biases += learning_rate * biases_gradient;
        // print weights and biases
        std::cout << "Weights after update: " << std::endl << network.layers[i].weights << std::endl;
        std::cout << "Biases after update: " << std::endl << network.layers[i].biases << std::endl << std::endl;

        // Update the gradients for the next layer
        // Note: This is a simplified version of backpropagation. For a more efficient and accurate implementation, consider using a more advanced optimization algorithm like Adam or RMSprop.
        gradients= gradients.transpose() * network.layers[i].weights;
        // gradients = gradients.transpose();
        // print gradients
        std::cout << "Gradients after update: " << gradients.transpose() << std::endl;
        // gradients = network.layers[i].weights.transpose() * gradients;
    }

    return error.array().square().sum();
}

// training the network
void train_neural_network(NeuralNetwork& network, const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            float error = backpropagation(network, inputs[i], targets[i], learning_rate);
            std::cout << "Epoch: " << epoch << ", Loss: " << error << std::endl;
        }
    }
}

// initializing the network
NeuralNetwork initialize_neural_network(int input_size, std::vector<int> hidden_sizes, int output_size) {
    NeuralNetwork network;

    // Initialize the first layer
    NeuralNetworkLayer layer;
    layer.weights = Eigen::MatrixXd::Random(hidden_sizes[0], input_size);
    layer.biases = Eigen::VectorXd::Random(hidden_sizes[0]);
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


// test the network
int main(){

    // Define the input and target data
    std::vector<Eigen::VectorXd> inputs = { Eigen::VectorXd::Random(2), Eigen::VectorXd::Random(2) };
    std::vector<Eigen::VectorXd> targets = { Eigen::VectorXd::Random(1), Eigen::VectorXd::Random(1) };

    // Initialize the neural network
    NeuralNetwork network = initialize_neural_network(2, { 4, 3 }, 1);

    // Train the neural network
    train_neural_network(network, inputs, targets, 1000, 0.1);

    // Test the neural network
    Eigen::VectorXd input = Eigen::VectorXd::Random(2);
    Eigen::VectorXd output = forward_propagation(network, input);
    std::cout << "Input: " << input.transpose() << ", Output: " << output.transpose() << std::endl;

    return 0;

}