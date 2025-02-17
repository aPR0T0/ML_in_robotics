#include<perceptron.hpp>

using namespace std;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<vector<uint8_t>> readMNISTImages(string filename, int &numImages, int &rows, int &cols) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2051) {
        cerr << "Invalid MNIST image file!" << endl;
        exit(1);
    }

    file.read((char*)&numImages, sizeof(numImages));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    numImages = reverseInt(numImages);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    vector<vector<uint8_t>> images(numImages, vector<uint8_t>(rows * cols));
    for (int i = 0; i < numImages; i++) {
        file.read((char*)images[i].data(), rows * cols);
    }

    file.close();
    return images;
}

vector<uint8_t> readMNISTLabels(string filename, int &numLabels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2049) {
        cerr << "Invalid MNIST label file!" << endl;
        exit(1);
    }

    file.read((char*)&numLabels, sizeof(numLabels));
    numLabels = reverseInt(numLabels);

    vector<uint8_t> labels(numLabels);
    file.read((char*)labels.data(), numLabels);

    file.close();
    return labels;
}


vector<Eigen::VectorXd> vector_to_eigen(int chunk_size, optional<vector<uint8_t>> data = std::nullopt, optional<vector<vector<uint8_t>>> data_2d = std::nullopt){
    // Result container: std::vector<Eigen::VectorXd>
    // print function was called
    // std::cout << "Data size: " << (data? data->size() : (data_2d? data_2d->size() * (*data_2d)[0].size() : 0)) << std::endl;
    // std::cout << "Chunk size: " << chunk_size << std::endl;
    
    
    vector<Eigen::VectorXd> eigen_vectors;
    if (chunk_size==1){
        std::cout<<"it goes here too"<<std::endl;
        // Compute how many Eigen::VectorXd will be created
        int num_vectors = (*data).size() / chunk_size;
        // Result container
        eigen_vectors.reserve(num_vectors);
        // Process data in chunks
        for (size_t i = 0; i < num_vectors; ++i) {
            Eigen::VectorXd vec(chunk_size);
            for (size_t j = 0; j < chunk_size; ++j) {
                vec[j] = static_cast<double>((*data)[i * chunk_size + j]);  // Safe conversion
            }
            eigen_vectors.push_back(vec);
        }
    }
    else{
        std::cout<<"it goes here at least"<<std::endl;

        eigen_vectors.reserve(data_2d->size());  // Reserve space for efficiency

        // Convert each std::vector<uint8_t> to Eigen::VectorXd
        for (const auto& row : (*data_2d)) {
            Eigen::VectorXd eigen_vector(row.size());
            for (size_t i = 0; i < row.size(); ++i) {
                eigen_vector[i] = static_cast<double>(row[i]);  // Safe conversion
            }
            eigen_vectors.push_back(eigen_vector);
        }
    }
    return eigen_vectors;
}
int find_max_index(const Eigen::VectorXd& vec) {
    return (vec.array() == vec.maxCoeff()).cast<int>().matrix().maxCoeff();
}
int main() {
    int numImages, rows, cols, numLabels;
    vector<vector<uint8_t>> images = readMNISTImages("dataset/train-images.idx3-ubyte", numImages, rows, cols);
    vector<uint8_t> labels = readMNISTLabels("dataset/train-labels.idx1-ubyte", numLabels);

    cout << "Loaded " << numImages << " images of size " << rows << "x" << cols << endl;
    cout << "Loaded " << numLabels << " labels" << endl;
    

    std::cout<<"I think network is not yet created"<<std::endl;
    // Define the input and target data
    std::vector<Eigen::VectorXd> inputs = vector_to_eigen(0, std::nullopt, images);
    std::vector<Eigen::VectorXd> targets = vector_to_eigen(1, labels, std::nullopt);

    // print the image size
    std::cout << "Image size: " << (int)images[0].size() << std::endl;

    // Initialize the neural network
    number_of_classes = 10;
    NeuralNetwork network = initialize_neural_network(images[0].size(), { (int)images[0].size()*2, (int)images[0].size(), 16 }, 10);
    size_t num_samples = 8;
    // Train the neural network
    train_neural_network(network, inputs, targets, 10, 0.1, num_samples);

    // print the network
    for (const auto& layer : network.layers) {
        std::cout << "Weights: " << std::endl << layer.weights << std::endl;
        std::cout << "Biases: " << std::endl << layer.biases << std::endl;
    }
    // Test the neural network
    vector<vector<uint8_t>> timages = readMNISTImages("dataset/t10k-images.idx3-ubyte", numImages, rows, cols);
    vector<uint8_t> tlabels = readMNISTLabels("dataset/t10k-labels.idx1-ubyte", numLabels);
    for (int i = 0; i < 100; i++){


        int index = rand() % timages.size();
        Eigen::VectorXd input = vector_to_eigen(0, std::nullopt, timages)[index];
        Eigen::VectorXd target = vector_to_eigen(1, tlabels, std::nullopt)[index];

        std::vector<Eigen::VectorXd> output = forward_propagation(network, input);
        std::cout << "Input: " << input.transpose() << ", Output: " << find_max_index(output.back()) << "Expected: "<< target.transpose()<<  std::endl;
    }
    return 0;
    
}
