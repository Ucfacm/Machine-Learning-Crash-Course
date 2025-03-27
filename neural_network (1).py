import numpy as np
import struct
import random
import kagglehub

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Path to dataset files:", path)

class Network(object):
    
    def __init__(self, layers):
        #Ex. layers = [784, 32, 10]
        
        # layers is a list of the number of neurons in each layer
        self.num_layers = len(layers)
        self.layers = layers
        
        # Initialize the biases and weights for each layer randomly
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])] 

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        #Derivative of the relu function.
        return np.where(z > 0, 1, 0)
    
    def softmax(self, z):
        # Shift z for numerical stability
        shifted_z = z - np.max(z)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=0)
    
    def forward_pass(self, input_data):
        # Start with the input data and go through each layer to get the final prediction
        activation = input_data
        
        # Feed Forward
        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):
            activation = self.relu(np.dot(weight, activation) + bias)
            
        # Return the final activation layer as a softmax
        return self.softmax(np.dot(self.weights[-1], activation) + self.biases[-1])

    def SGD(self, training_data, epochs, mini_batch_size, learn_rate, test_data=None):
        # Train with or without test data checking, makes training faster
        if test_data: 
            num_test_examples = len(test_data)
        num_train_examples = len(training_data)
        
        # Go through each epoch and update the model with each iteration
        for epoch in range(epochs):
            random.shuffle(training_data)
        
            # Create mini batches from the training data to speed up training
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, num_train_examples, mini_batch_size)]
        
            for mini_batch in mini_batches:
                self.update_model(mini_batch, learn_rate)
        
            # Check the model accuracy after each epoch if test data exists
            if test_data:
                print(f"Epoch {epoch}: {self.test(test_data)} / {num_test_examples}")
            else:
                print(f"Epoch {epoch} complete")

    def update_model(self, train_set, learn_rate):
        # Create arrays filled with zeros to store the gradients for each layer (nabla -> ∇)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Go through each training example in the batch
        for x, y in train_set:
            # For each training example, calculate the gradients for each layer
            delta_nabla_b, delta_nabla_w = self.back_propagate(x, y)
            
            # Sum all the gradients of all the training examples
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Apply the average of the gradient to the current weights and biases   
        self.weights = [w - (learn_rate/len(train_set)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learn_rate/len(train_set)) * nb for b, nb in zip(self.biases, nabla_b)]
        
        # NOTE: THE LEARNING RATE IS DIVIDED INSTEAD OF THE GRADIENTS TO REDUCE CALCULATION TIME

    def back_propagate(self, x, y):
        # Create arrays filled with zeros to store the gradients for each layer
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Current activation layer is the input layer        
        cur_layer_activation = x 
        
        # list to store all the activations, layer by layer
        activations = [x]
        
        # list to store all the z values for each layer
        z_list = []
        
        # Forward Pass
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, cur_layer_activation) + b
            z_list.append(z)
            cur_layer_activation = self.relu(z)
            activations.append(cur_layer_activation)
    
        z = np.dot(self.weights[-1], cur_layer_activation) + self.biases[-1]
        z_list.append(z)
        cur_layer_activation = self.softmax(z)
        activations.append(cur_layer_activation)
            
        # Backward Pass

        # For the output layer with softmax, we can directly use (output - target)
        delta = activations[-1] - y
        
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Back propagate to the previous layers, working backwards from the output layer
        for i in range(2, self.num_layers):
            # Calculate the error in the current layer using the error in the next layer
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * self.relu_prime(z_list[-i])
            
            # Change the gradients for the biases and weights based on the error calculated
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return (nabla_b, nabla_w)

    def test(self, test_data):
        # Feed forward the test data and use the strongest activation as the prediction
        test_results = [(np.argmax(self.forward_pass(x)), np.argmax(y)) for (x, y) in test_data]
        
        # Return the number of correct predictions
        return sum(int(x == y) for (x, y) in test_results)

def read_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images

def read_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def import_data():
    # File paths for the MNIST dataset
    training_images_filepath = path + '/train-images.idx3-ubyte'
    training_labels_filepath = path + '/train-labels.idx1-ubyte'
    test_images_filepath = path + '/t10k-images.idx3-ubyte'
    test_labels_filepath = path + '/t10k-labels.idx1-ubyte'
    
    # Read the images and labels
    x_train = read_images(training_images_filepath)
    y_train = read_labels(training_labels_filepath)
    x_test = read_images(test_images_filepath)
    y_test = read_labels(test_labels_filepath)
    
    # Normalize the images to the range [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    x_train = [x.reshape(784, 1) for x in x_train]
    x_test = [x.reshape(784, 1) for x in x_test]
    
    def to_np_array(y):
        arr = np.zeros((10, 1))
        arr[y] = 1.0
        return arr
    
    # Convert labels to one-hot encoded vectors
    y_train_encoded = [to_np_array(y) for y in y_train]
    y_test_encoded = [to_np_array(y) for y in y_test]
    
    return (x_train, y_train_encoded), (x_test, y_test_encoded)

def main():
    # Import the data
    (x_train, y_train), (x_test, y_test) = import_data()
    
    # Create the network
    net = Network([784, 128, 10])
    
    # Train the network
    net.SGD(list(zip(x_train, y_train)), 10, 10, 0.1, test_data=list(zip(x_test, y_test)))
    
    # Test the network
    print(f"Test data: {net.test(list(zip(x_test, y_test)))} / {len(x_test)}")
    
if __name__ == "__main__":
    main()