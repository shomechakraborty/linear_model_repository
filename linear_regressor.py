import numpy as np, matplotlib.pyplot as plt

class LinearRegressor:
    def __init__(self, input_size, epochs = 100, learning_rate = 0.01):
        self.input_size = input_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = np.array([0.0 for i in range(self.input_size + 1)])

    def run_model(self, input_training_data):
        transformed_inputs = np.concatenate([np.array([[1 for i in range(len(input_training_data.T[0]))]]), input_training_data.T])
        outputs = np.dot(self.weights, transformed_inputs)
        return transformed_inputs, outputs

    def run_evaluation(self, input_training_data, input_validation_data, target_training_data, target_validation_data):
        _, training_outputs = self.run_model(input_training_data)
        training_loss = target_training_data - training_outputs
        training_mse = np.sum(pow(training_loss, 2)) / len(training_loss)
        _, validation_outputs = self.run_model(input_validation_data)
        validation_loss = target_validation_data - validation_outputs
        validation_mse = np.sum(pow(validation_loss, 2)) / len(validation_loss)
        return training_mse, validation_mse

    def train_model(self, input_data, target_data, val_prop = 0.8, show = False):
        training_mses_across_epochs = []
        validation_mses_across_epochs = []
        input_training_data = np.array(input_data)[:round(len(input_data) * val_prop)]
        input_validation_data = np.array(input_data)[round(len(input_data) * val_prop):]
        target_training_data = np.array(target_data)[:round(len(target_data) * val_prop)]
        target_validation_data = np.array(target_data)[round(len(target_data) * val_prop):]
        for i in range(self.epochs):
            random_training_indices = np.random.choice(len(input_training_data), len(input_training_data), replace = False)
            input_training_data = input_training_data[random_training_indices]
            target_training_data = target_training_data[random_training_indices]
            transformed_inputs, outputs = self.run_model(np.array(input_training_data))
            loss = target_training_data - outputs
            gradients = np.array([np.sum(np.array([loss[j] * -1 * transformed_inputs.T[j][i] for j in range(len(loss))])) * (2 / len(loss)) for i in range(len(self.weights))])
            velocities = self.learning_rate * gradients
            self.weights -= velocities
            training_mse, validation_mse = self.run_evaluation(input_training_data, input_validation_data, target_training_data, target_validation_data)
            training_mses_across_epochs.append(training_mse)
            validation_mses_across_epochs.append(validation_mse)
        if show:
            plt.plot(training_mses_across_epochs)
            plt.plot(validation_mses_across_epochs)
            plt.show()

    def predict(self, input_data):
        _, outputs = self.run_model(np.array(input_data))
        return outputs
