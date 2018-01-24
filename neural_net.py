import os
import tensorflow as tf
import numpy as np

from cnnbot.common import PLANET_MAX_NUM, PER_PLANET_FEATURES

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


# Normalize planet features within each frame.
def normalize_input(input_data):

    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


class ConvNeuralNet(object):

    def __init__(self, cached_model=None, seed=None):

        self._graph = tf.Graph()

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, 64, 64, 3))

            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]

            self._target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                       shape=(None, PLANET_MAX_NUM))

            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            input_flattened_frames = tf.reshape(self._features, [-1, 64, 64, 3])

            #Convolutional Layer 1 (32 5X5 filters, activation=ReLU) and Max Pooling Layer 1 (2x2, stride=2)
            conv_first_layer = tf.layers.conv2d(inputs=input_flattened_frames, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.leaky_relu(alpha=0.3))

            pool_first_layer = tf.layers.max_pooling2d(inputs=conv_first_layer, pool_size=[2, 2], strides=2)

            #Convolutional Layer 2 (32 5X5 filters, activation=ReLU) and Max Pooling Layer 2 (2x2, stride=2)
            conv_second_layer= tf.layers.conv2d(inputs=pool_first_layer, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.leaky_relu(alpha=0.3))

            pool_second_layer = tf.layers.max_pooling2d(inputs=conv_second_layer, pool_size=[2, 2], strides=2)

            # Dense Layer 1 with dropout regularization
            pool2_flat = tf.reshape(pool_second_layer, [-1, 16 * 16 * 32])
            dense_first_layer = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu)
            dropout_first = tf.nn.dropout(x=dense_first_layer, keep_prob=0.7, seed=0)

            #Dense Layer 2 with dropout regularization
            dense_second_layer=tf.layers.dense(inputs=dropout_first, units=500, activation=tf.nn.relu)
            dropout_second = tf.nn.dropout(x=dense_second_layer, keep_prob=0.7, seed=0)

            #Dense Layer 3 with dropout regularization
            dense_third_layer=tf.layers.dense(inputs=dropout_second, units=500, activation=tf.nn.relu)
            dropout_third = tf.nn.dropout(x=dense_third_layer, keep_prob=0.7, seed=0)

            #Dense Layer 4 with dropout regularization
            dense_fourth_layer=tf.layers.dense(inputs=dropout_third, units=500, activation=tf.nn.relu)
            dropout_fourth = tf.nn.dropout(x=dense_fourth_layer, keep_prob=0.7, seed=0)

            #Dense Layer 5 with dropout regularization
            dense_fifth_layer=tf.layers.dense(inputs=dropout_fourth, units=500, activation=tf.nn.relu)
            dropout_fifth = tf.nn.dropout(x=dense_fifth_layer, keep_prob=0.7, seed=0)

            # Logits Layer
            logits = tf.layers.dense(inputs=dropout_fifth, units= PLANET_MAX_NUM)

            #Generate predictions
            self._prediction_normalized = tf.nn.softmax(logits)

            #Calculate loss
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()


            file_writer = tf.summary.FileWriter("[add a log directory and paste the path]", tf.get_default_graph())


            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

            file_writer.close()

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: input_data,
                                               self._target_distribution: expected_output_data})
        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self._session.run(self._prediction_normalized,
                                 feed_dict={self._features: np.array([input_data])})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: input_data,
                                            self._target_distribution: expected_output_data})

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)
