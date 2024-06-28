//Implementing XOR Gate using a single hidden layer neural network with 4 neurons in C from scratch

// I know a C code is not really supposed to be in a pytorch tutorial repo but inspired by Andrej Karpathy's work, I wanted 
// to build just a simple neural net in a simple C script. That is the main motivation here. 

//For a learning rate of 0.1, I couldn't get a decent accuracy until around 10000 epochs.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1
#define NUM_EPOCHS 10000

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of activation function
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

int main() {
    // Training data
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected_outputs[4][1] = {{0}, {1}, {1}, {0}};

    // Neural network parameters
    double hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    double hidden_bias[HIDDEN_SIZE];
    double output_weights[HIDDEN_SIZE][OUTPUT_SIZE];
    double output_bias[OUTPUT_SIZE];

    // Initialize weights and biases randomly
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    output_bias[0] = ((double)rand() / RAND_MAX) * 2 - 1;

    // Train the neural network
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double total_loss = 0;
        int correct_count = 0;

        for (int i = 0; i < 4; i++) {
            // Forward propagation
            double hidden_layer[HIDDEN_SIZE];
            double output_layer[OUTPUT_SIZE];

            // Calculate hidden layer
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double sum = 0;
                for (int k = 0; k < INPUT_SIZE; k++) {
                    sum += inputs[i][k] * hidden_weights[k][j];
                }
                hidden_layer[j] = sigmoid(sum + hidden_bias[j]);
            }

            // Calculate output layer
            double output_sum = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output_sum += hidden_layer[j] * output_weights[j][0];
            }
            output_layer[0] = sigmoid(output_sum + output_bias[0]);

            // Calculate loss
            double output_error = expected_outputs[i][0] - output_layer[0];
            total_loss += output_error * output_error;

            // Calculate accuracy
            if ((output_layer[0] >= 0.5 && expected_outputs[i][0] == 1) || 
                (output_layer[0] < 0.5 && expected_outputs[i][0] == 0)) {
                correct_count++;
            }

            // Backpropagation
            double output_delta = output_error * sigmoid_derivative(output_layer[0]);

            double hidden_errors[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_errors[j] = 0;
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    hidden_errors[j] += output_weights[j][k] * output_delta;
                }
            }

            double hidden_deltas[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights and biases
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int k = 0; k < INPUT_SIZE; k++) {
                    hidden_weights[k][j] += LEARNING_RATE * inputs[i][k] * hidden_deltas[j];
                }
                hidden_bias[j] += LEARNING_RATE * hidden_deltas[j];
            }

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output_weights[j][0] += LEARNING_RATE * hidden_layer[j] * output_delta;
            }
            output_bias[0] += LEARNING_RATE * output_delta;
        }

        // Print loss and accuracy every 1000 epochs
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%\n", epoch, total_loss / 4, (correct_count / 4.0) * 100);
        }
    }

    // Test the trained network
    printf("Testing the accuracy of XOR neural network:\n");
    for (int i = 0; i < 4; i++) {
        double hidden_layer[HIDDEN_SIZE];
        double output_layer[OUTPUT_SIZE];

        // Calculate hidden layer
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = 0;
            for (int k = 0; k < INPUT_SIZE; k++) {
                sum += inputs[i][k] * hidden_weights[k][j];
            }
            hidden_layer[j] = sigmoid(sum + hidden_bias[j]);
        }

        // Calculate output layer
        double output_sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_sum += hidden_layer[j] * output_weights[j][0];
        }
        output_layer[0] = sigmoid(output_sum + output_bias[0]);

        printf("Input: [%.0f, %.0f], Output: %.2f\n", inputs[i][0], inputs[i][1], output_layer[0]);
    }

    return 0;
}
