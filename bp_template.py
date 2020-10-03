import numpy as np
import time

# neural network class definition
class neuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        The network consists of three types of layers: input layer(784 nodes), hidden layers and output layer(10 nodes).
        You can define the number of hidden nodes and layers you want.
        '''
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.inputs = None
        self.hidden_outputs = None
        self.final_outputs = None
        self.lr = learningrate

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.inodes, self.hnodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.onodes))

        self.activation_function = lambda x: 1./(1 + np.exp(-x))
        #???????????????????????????//
        pass

    def forward(self, inputs_list):
        '''
        forward the neural network
        '''
        self.inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.inputs.T, self.wih)#?????????????????????
        self.hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.hidden_outputs, self.who)
        self.final_outputs = self.activation_function(final_inputs)
        pass

    def Backpropagation(self, targets_list):
        '''
        propagate backword
        '''
        targets = np.array(targets_list, ndmin=2)

        output_errors = targets - self.final_outputs
        g_temp = self.final_outputs[0] * (1 - self.final_outputs[0]) * output_errors[0]
        g = np.array(g_temp, ndmin=2).T
        self.who = self.who + self.lr * np.dot(self.hidden_outputs.T, g.T)

        #def cal_temp(h):
        #    ans = 0
        #    for i in range(0, self.onodes):
        #        ans += self.who[h][i] * g[i]
        #    return ans

        #for i in range(0, self.hnodes):
        #    e_temp[i] = self.hidden_outputs[0][i] * (1 - self.hidden_outputs[0][i]) * cal_temp(i)
        e_temp = self.hidden_outputs[0] * (1 - self.hidden_outputs[0]) * np.dot(self.who, g).T
        e = np.array(e_temp, ndmin=2)
        temp2 = self.lr * np.dot(self.inputs, e)
        self.wih = self.wih + temp2
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
        temp1 = None
        temp2 = None
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


            #if i == 0:
            #    temp1 = net.wih
            #    temp2 = net.who
            #else:
            #    if (net.wih == temp1).all():
            #        print("", end="")
            #    else:
            #        print("111")
            #    if (net.who == temp2).all():
            #        print("", end="")
            #    else:
            #        print("222change!")
            #    temp1 = net.wih
            #    temp2 = net.who


            # print error
            if (i + 1) % 5000 == 0:
                print("epochs: {}, error: {}".format(e, error / 1000))
                error = 0
        test_net(net, training_data_list)

def test_net(net, data_list):
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
    hidden_nodes = 200 # you can define the numbers of nodes you want and more hidden layers
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