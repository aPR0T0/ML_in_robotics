// Write the base template for the eigen library for a multilayer perceptron
#include <perceptron.hpp>

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
        // std::cout << layer.weights.rows() << " " << layer.weights.cols() << " " <<std::endl;
        // output = sigmoid(layer.weights.transpose() * output + layer.biases); // apply the sigmoid activation function
        output.push_back(sigmoid(layer.weights * output.back() + layer.biases)); // input is the initial values for the perceptron
        // output for each layer
    }
    // print the output
    return output;
}

// create an array of target equivalent of size of the output layer

Eigen::VectorXd change_target_dimesions(const Eigen::VectorXd& labels, int nums) {
    Eigen::VectorXd one_hot = Eigen::VectorXd::Zero(nums);  // Initialize with zeros

    int index = static_cast<int>(labels[0]);  // Extract integer label
    if (index >= 0 && index < nums) {
        one_hot(index) = 1.0;  // Set corresponding index to 1
    } else {
        std::cerr << "Error: Label index " << index << " is out of range!" << std::endl;
    }

    return one_hot;
}

// Define the backpropagation function
float backpropagation(NeuralNetwork& network, const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learning_rate) {

    // Backpropagation
    std::vector<Eigen::VectorXd> output = forward_propagation(network, input);
    // print the target
    Eigen::VectorXd targetted = change_target_dimesions(target, number_of_classes); // num_classes is 10
    // std::cout << "target: " << target.transpose() << std::endl;
    Eigen::MatrixXd hidden_error = targetted - output.back();
    // print out the error
    // std::cout << "hidden_error: " << hidden_error.transpose() << std::endl;
    output.pop_back();
    // Calculate the hidden_errors
    Eigen::MatrixXd hidden_errors;
    
    for (int i = network.layers.size() - 1; i > 0; i--) {
        // print output 
        // std::cout << sigmoid_derivative(output.back()) << std::endl;
        // std::cout << "hidden: " << hidden_error << std::endl;
        // here the output is basically all the neurons as the output from the current hidden layer
        hidden_errors = output.back() * hidden_error.transpose();
        // std::cout<<"Something is happening"<<std::endl;
        // print hidden_errors
        // std::cout << "Hidden Errors: " << hidden_error.transpose() << std::endl;
        // update weights and biases    

        network.layers[i].weights += learning_rate * hidden_errors.transpose();
        network.layers[i].biases += learning_rate * hidden_error;
        // std::cout << "Weights: " << std::endl << network.layers[i].biases << std::endl;
        // std::cout<< " output dims " << sigmoid_derivative(output.back().transpose()) << std::endl;
        // std::cout << std::endl;

        // And here the output means the output of the hidden layer
        hidden_error =  hidden_errors * network.layers[i].weights * sigmoid_derivative(output.back());

        output.pop_back();
        // std::cout << "hidden Errors: " << hidden_error.transpose() << std::endl;

    }
    return hidden_errors.array().square().sum();
}

// Mutex for shared resources (if needed)
std::mutex mutx;

void train_batch(NeuralNetwork& network, const std::vector<Eigen::VectorXd>& inputs, 
    const std::vector<Eigen::VectorXd>& targets, double learning_rate, size_t start, size_t end) {

    for (int i = start; i < end; ++i) {

        std::cout<< "Training"<< std::endl;

        float error = backpropagation(network, inputs[i], targets[i], learning_rate);

        std::lock_guard<std::mutex> lock(mutx);
        std::cout << "Thread " << std::this_thread::get_id() << " processed index " << i << ", Loss: " << error << std::endl;
        // release mutex
        // lock.unlock();
    }
}

// training the network
void train_neural_network(NeuralNetwork& network, const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs, double learning_rate, size_t num_samples) {
    
    int num_threads = 4;
    // print current number of threads
    std::cout << "Currently available Number of threads: " << num_threads << std::endl;

    // size_t num_samples = inputs.size();
    // size_t num_samples = 80;
    size_t chunk_size = num_samples / num_threads;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size * (epoch + 1);
            size_t end = start + chunk_size;
            threads.emplace_back(train_batch, std::ref(network), std::cref(inputs), std::cref(targets), learning_rate, start, end);
        }

        // Join threads
        for (auto& th : threads) th.join();

        std::cout << "Epoch " << epoch << " completed." << std::endl;
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
// int main(){

//     // Define the input and target data
//     std::vector<Eigen::VectorXd> inputs = { Eigen::VectorXd::Random(2), Eigen::VectorXd::Random(2) };
//     std::vector<Eigen::VectorXd> targets = { Eigen::VectorXd::Random(1), Eigen::VectorXd::Random(1) };

//     // Initialize the neural network
//     NeuralNetwork network = initialize_neural_network(2, { 4, 3, 6, 8 }, 1);

//     // Train the neural network
//     train_neural_network(network, inputs, targets, 1000, 0.1);

//     // print the network
//     for (const auto& layer : network.layers) {
//         std::cout << "Weights: " << std::endl << layer.weights << std::endl;
//         std::cout << "Biases: " << std::endl << layer.biases << std::endl;
//     }
//     // Test the neural network
//     Eigen::VectorXd input = Eigen::VectorXd::Random(2);
//     std::vector<Eigen::VectorXd> output = forward_propagation(network, input);
//     std::cout << "Input: " << input.transpose() << ", Output: " << output.back().transpose() << std::endl;

//     return 0;

// }