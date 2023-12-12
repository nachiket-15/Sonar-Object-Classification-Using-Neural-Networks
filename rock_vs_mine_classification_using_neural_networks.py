
import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Forward propagation
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            predicted_output = self.sigmoid(output_layer_input)

            # Backward propagation
            error = y - predicted_output
            output_delta = error * self.sigmoid_derivative(predicted_output)

            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta) * self.learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

            # Print epoch and training accuracy every 100 epochs
            if epoch % 100 == 0:
                X_train_prediction = self.predict(X)
                training_data_accuracy = np.mean(X_train_prediction == y.reshape(-1, 1))
                print(f'Epoch {epoch}, Training Accuracy: {training_data_accuracy*100:.2f} %')




    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        predicted_output = self.sigmoid(output_layer_input)

        return np.round(predicted_output)

#Read input data
sonar_data = pd.read_csv('/content/sonar_data.csv', header=None)

# Separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]





# Convert 'R' to 0 and 'M' to 1 in the target variable
Y = Y.map({'R': 0, 'M': 1})

# Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Model Training - Neural Network
input_size = X_train.shape[1]
hidden_size = 24
output_size = 1
learning_rate = 0.01
epochs = 2000



model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, epochs)
model.fit(X_train.values, Y_train.values.reshape(-1, 1))  # Ensure Y_train is of shape (m, 1)

# Model Evaluation
# Accuracy on training data
X_train_prediction = model.predict(X_train.values)
training_data_accuracy = np.mean(X_train_prediction == Y_train.values.reshape(-1, 1))
print(f'\nAccuracy on training data : , {training_data_accuracy*100:.2f} %')






# Accuracy on test data
X_test_prediction = model.predict(X_test.values)
test_data_accuracy = np.mean(X_test_prediction == Y_test.values.reshape(-1, 1))
print(f'Accuracy on test data : , {test_data_accuracy*100:.2f} %')







# Assuming X_test is your feature dataset for testing

choice=int(input("Enter the row for testint ? (between 0-20)"))

test_input_row = X_test.iloc[choice, :]
print("\nSome of starting attributes for selected choice from dataset are:\n")
print(test_input_row.head(5))

# print(test_input_row)
# Reshape the input data as we are predicting for one instance
test_input_reshaped = test_input_row.values.reshape(1, -1)









prediction = model.predict(test_input_reshaped)
print(prediction)

print("\n")

if prediction[0] == 1:
    print('The object is a MINE')
else:
    print('The object is a ROCK')

