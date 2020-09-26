import numpy as np
import time

# neural network class definition
class neuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        The network consists of three layers: input layer, hidden layer and output layer.
        Here defined these layers.
        '''
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Some parameters that will be used next
        self.inputs = None          # input data
        self.hidden_outputs = None  # the output of hidden layer
        self.final_outputs = None   # the output of output layer
        self.lr = learningrate      # learning rate
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))    # init the weight of input layers to hidden layers
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))    # init the weight of hidden layers to output layers
        
        # activation function is the sigmoid function
        # self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function = lambda x: 1. / (1 + np.exp(-x))
        
        pass

    def forward(self, inputs_list):
        '''
        forward the neural network
        '''
        # convert inputs list to 2d array
        self.inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, self.inputs)
        # calculate the signals emerging from hidden layer
        self.hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, self.hidden_outputs)
        # calculate the signals emerging from final output layer
        self.final_outputs = self.activation_function(final_inputs)
        
        pass

    def Backpropagation(self, targets_list):
        '''
        propagate backword
        '''

        targets = np.array(targets_list, ndmin=2).T
        # output layer error is the (target - actual)
        output_errors = targets - self.final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * self.final_outputs * (1.0 - self.final_outputs)), np.transpose(self.hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * self.hidden_outputs * (1.0 - self.hidden_outputs)), np.transpose(self.inputs))
        
        return np.sum(output_errors ** 2)


def train_net(net, training_data_list):
    '''
    Train the neural network
    '''

    # train the neural network

    # epochs is the number of times the training data set is used for training
    epochs = 10

    for e in range(epochs):
        # go through all records in the training data set
        error = 0
        for i in range(len(training_data_list)):
            record = training_data_list[i]
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99

            # Forward network and propagate backward
            net.forward(inputs)
            error += net.Backpropagation(targets)

            # print error
            if (i + 1) % 5000 == 0:
                print("epochs: {}, error: {}".format(e, error / 1000))
                error = 0
        test_net(net, training_data_list)

def test_net(net, data_list, record_time = False):
    '''
    Test network accuracy with data_list
    During training, the accuracy is the result of testing using the training set.
    During testing, the accuracy is the result of testing using the testing set.
    '''
    
    cnt = 0
    for record in data_list:
        all_values = record.split(',')
        img_array = np.asfarray(all_values[1:])

        # reshape from 28x28 to list of 784 values, invert values
        img_data  = img_array.reshape(784)
        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01

        # forward the network
        net.forward(img_data)
        outputs = net.final_outputs

        # the index of the highest value corresponds to the label
        ans = np.argmax(outputs)
        if int(ans) == int(all_values[0]):
            cnt += 1

    print("Accuracy: {:.2f}".format(cnt / len(data_list)))


if __name__ == "__main__":

    # record start time
    start_time = time.time()

    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate
    learning_rate = 0.1

    # create instance of neural network
    net = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # get train data
    training_data_file = open("./mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    # print(len(training_data_list))
    training_data_file.close()

    # record start training time
    start_training_time = time.time()
    # start training
    train_net(net, training_data_list)
    print("Training time: {:.3f}s".format(time.time() - start_training_time))

    # load image data from png files into an array
    test_data_file = open("./mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    # print(len(test_data_list))
    # print(test_data_list[0])
    test_data_file.close()

    # record start testing time
    start_testing_time = time.time()
    # Use the test set to test the final recognition accuracy
    test_net(net, test_data_list)
    print("Testing time: {:.3f}s".format(time.time() - start_testing_time))

    # Save weight parameters
    np.save('./who.npy', net.who)
    np.save('./whi.npy', net.wih)

    # Ending time
    end_time = time.time()
    print("Program running time: {:.3f}s".format(time.time() - start_time))