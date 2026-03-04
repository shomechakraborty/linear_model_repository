import numpy as np, pandas as pd, matplotlib.pyplot as plt

class LinearClassifier:
    def __init__(self, input_size, num_classes, epochs = 100, learning_rate = 0.01):
        self.input_size = input_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = np.array([[0.0 for j in range(self.input_size + 1)] for i in range(self.num_classes)])

    def run_model(self, input_training_data):
        transformed_inputs = np.concatenate([np.array([[1 for i in range(len(input_training_data.T[0]))]]), input_training_data.T])
        weighted_sums = np.dot(self.weights, transformed_inputs)
        activations = np.exp(weighted_sums) / np.sum(np.exp(weighted_sums), axis = 0)
        outputs = self.classes[np.argmax(activations, axis = 0)]
        return transformed_inputs, weighted_sums, activations, outputs

    def run_evaluation(self, input_training_data, input_validation_data, target_training_data, target_validation_data):
        _, _, _, training_outputs = self.run_model(input_training_data)
        training_targets = self.classes[np.argmax(target_training_data, axis = 1)]
        training_accuracy = np.sum(np.equal(training_outputs, training_targets)) / len(training_targets)
        _, _, _, validation_outputs = self.run_model(input_validation_data)
        validation_targets = self.classes[np.argmax(target_validation_data, axis = 1)]
        validation_accuracy = np.sum(np.equal(validation_outputs, validation_targets)) / len(validation_targets)
        return training_accuracy, validation_accuracy

    def train_model(self, input_data, target_data, val_prop = 0.8, show = False):
        training_accuracies_across_epochs = []
        validation_accuracies_across_epochs = []
        input_training_data = np.array(input_data)[:round(len(input_data) * val_prop)]
        input_validation_data = np.array(input_data)[round(len(input_data) * val_prop):]
        target_encoders = np.array(pd.get_dummies(pd.Series(np.array(target_data)), dtype = float))
        target_training_data = target_encoders[:round(len(target_data) * val_prop)]
        target_validation_data = target_encoders[round(len(target_data) * val_prop):]
        self.classes = np.array(pd.get_dummies(pd.Series(np.array(target_data)), dtype = float).columns)
        for i in range(self.epochs):
            random_training_indices = np.random.choice(len(input_training_data), len(input_training_data), replace = False)
            input_training_data = input_training_data[random_training_indices]
            target_training_data = target_training_data[random_training_indices]
            transformed_inputs, weighted_sums, activations, outputs = self.run_model(np.array(input_training_data))
            loss_derivative_with_respect_to_activation = np.array([[activations[i][j] - target_training_data[j][i] for j in range(len(activations[i]))] for i in range(len(activations))])
            activation_derivative_with_respect_to_weighted_sum = np.array([[activations[i][j] * (1 - activations[i][j]) for j in range(len(activations[i]))] for i in range(len(activations))])
            loss_derivative_with_respect_to_weighted_sum = loss_derivative_with_respect_to_activation * activation_derivative_with_respect_to_weighted_sum
            loss_derivative_with_respect_to_weight = np.array([[loss_derivative_with_respect_to_weighted_sum[i][j] * transformed_inputs.T[j] for j in range(len(loss_derivative_with_respect_to_weighted_sum[i]))] for i in range(len(loss_derivative_with_respect_to_weighted_sum))])
            gradients = np.array([np.average(loss_derivative_with_respect_to_weight[i], axis = 0) for i in range(len(loss_derivative_with_respect_to_weight))])
            velocities = self.learning_rate * gradients
            self.weights -= velocities
            training_accuracy, validation_accuracy = self.run_evaluation(input_training_data, input_validation_data, target_training_data, target_validation_data)
            training_accuracies_across_epochs.append(training_accuracy)
            validation_accuracies_across_epochs.append(validation_accuracy)
        if show:
            plt.plot(training_accuracies_across_epochs)
            plt.plot(validation_accuracies_across_epochs)
            plt.show()

    def predict(self, input_data):
        _, _, _, outputs = self.run_model(np.array(input_data))
        return outputs
