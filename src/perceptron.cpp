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
std::vector<Eigen::VectorXd> forward_propagation(const NeuralNetwork& network, const Eigen::VectorXd& input) {
    std::vector<Eigen::VectorXd> output; // Initialized the output vector
    output.push_back(input); // Input layer is the initial values for the perceptron

    for (const auto& layer : network.layers) {
        // print the dimensions of the layer weights and the output vector
        std::cout << layer.weights.rows() << " " << layer.weights.cols() << " " <<std::endl;
        // output = sigmoid(layer.weights.transpose() * output + layer.biases); // apply the sigmoid activation function
        output.push_back(sigmoid(layer.weights * output.back() + layer.biases)); // input is the initial values for the perceptron
        // output for each layer
    }

    return output;
}


// Define the backpropagation function
float backpropagation(NeuralNetwork& network, const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learning_rate) {

    // Backpropagation
    std::vector<Eigen::VectorXd> output = forward_propagation(network, input);
    Eigen::MatrixXd hidden_error = target - output.back();
    // print out the error
    output.pop_back();
    // Calculate the hidden_errors
    
    for (int i = network.layers.size() - 1; i > 0; i--) {
        // print output 
        // std::cout << sigmoid_derivative(output.back()) << std::endl;
        std::cout << "hidden: " << hidden_error << std::endl;
        // here the output is basically all the neurons as the output from the current hidden layer
        Eigen::MatrixXd hidden_errors = output.back() * hidden_error.transpose();

        // print hidden_errors
        std::cout << "Hidden Errors: " << hidden_errors.transpose() << std::endl;
        // update weights and biases    

        network.layers[i].weights += learning_rate * hidden_errors.transpose();
        network.layers[i].biases += learning_rate * hidden_error;
        // std::cout << "Weights: " << std::endl << network.layers[i].biases << std::endl;
        std::cout<< " output dims " << sigmoid_derivative(output.back().transpose()) << std::endl;
        std::cout << std::endl;

        // And here the output means the output of the hidden layer
        hidden_error =  hidden_errors * network.layers[i].weights * sigmoid_derivative(output.back());

        output.pop_back();
        std::cout << "hidden Errors: " << hidden_error.transpose() << std::endl;

    }

    return hidden_error.array().square().sum();
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
    NeuralNetwork network = initialize_neural_network(2, { 4, 3, 6, 8 }, 1);

    // Train the neural network
    train_neural_network(network, inputs, targets, 1000, 0.1);

    // print the network
    for (const auto& layer : network.layers) {
        std::cout << "Weights: " << std::endl << layer.weights << std::endl;
        std::cout << "Biases: " << std::endl << layer.biases << std::endl;
    }
    // Test the neural network
    Eigen::VectorXd input = Eigen::VectorXd::Random(2);
    std::vector<Eigen::VectorXd> output = forward_propagation(network, input);
    std::cout << "Input: " << input.transpose() << ", Output: " << output.back().transpose() << std::endl;

    return 0;

}