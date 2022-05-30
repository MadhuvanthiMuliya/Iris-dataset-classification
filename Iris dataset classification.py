# Import Libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Load Iris Dataset
iris = datasets.load_iris()
Features = iris.data
labels = iris.target
# Split dataset into training and test sets
F_train, F_test, l_train, l_test = train_test_split(Features, labels, test_size=0.2, stratify=labels, random_state=0)
# One-hot encoding of labels
l_train_o = tf.keras.utils.to_categorical(l_train, num_classes=3)
l_test_o = tf.keras.utils.to_categorical(l_test, num_classes=3)
print('No. of Features in Training Set -> ', F_train.shape)
print('No. of Labels in Training Set -> ', l_train_o.shape)
print('No. of Features in Test Set -> ', F_test.shape)
print('No. of Labels in Test Set -> ', l_test_o.shape)


class Network:

    def __init__(self):
        self.n_inputs = num_inputs
        self.n_hidden = num_hidden
        self.n_class = num_class
        self.lr = learning_rate

        self.initializer = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.all_parameters = []
        self.optimizer = tf.optimizers.SGD(self.lr)
        self.initialize_network()

    def initialize_network(self):
        weight_shapes = [[self.n_inputs, self.n_hidden], [self.n_hidden, self.n_hidden], [self.n_hidden, self.n_class]]
        m = len(weight_shapes)
        for i in range(m):
            param = tf.Variable(initial_value=self.initializer(weight_shapes[i]), name='Weight {}'.format(i),
                                trainable=True, dtype=tf.float32)
            self.all_parameters.append(param)

    def forward_network(self, inputs):
        l_inputs = tf.cast(inputs, dtype=tf.float32)
        l1 = tf.cast(tf.nn.relu(tf.matmul(l_inputs, self.all_parameters[0])), dtype=tf.float32)
        l2 = tf.cast(tf.nn.relu(tf.matmul(l1, self.all_parameters[1])), dtype=tf.float32)
        l3 = tf.cast(tf.nn.softmax(tf.matmul(l2, self.all_parameters[2])), dtype=tf.float32)
        return l3


class TrainModel:
    def __int__(self):
        self.x_train = F_train
        self.y_train = l_train_o

    def batch_split(self, b_size):
        # Return a total of 'size random samples and labels.
        idx = np.arange(0, F_train.shape[0])
        np.random.shuffle(idx)
        idx = idx[:b_size]
        x_shuffle = [F_train[i] for i in idx]
        y_shuffle = [l_train_o[i] for i in idx]

        return np.asarray(x_shuffle), np.asarray(y_shuffle)

    @tf.function
    def learn(self, x_shuff, y_shuff):
        y_shuff = tf.cast(y_shuff, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            output_pred = model.forward_network(x_shuff)
            loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=y_shuff, logits=output_pred)

        grads = tape.gradient(loss_, model.all_parameters)
        model.optimizer.apply_gradients(zip(grads, model.all_parameters))

        return loss_


# Hyperparameters
num_inputs = 4
num_hidden = 10
num_class = 3
learning_rate = 0.005
batch_size = 40

# Initialize Model
model = Network()
Agent = TrainModel()

total_episodes = 10000
ep_loss_list = []

for ep in range(total_episodes):
    feature_batch, label_batch = Agent.batch_split(batch_size)
    loss = Agent.learn(feature_batch, label_batch)
    ep_loss = sum(loss)/batch_size
    ep_loss_list.append(ep_loss)

    if ep % 100 == 0:
        print('Epoch: ', ep, ', Loss per Epoch: ', ep_loss)


plt.plot(ep_loss_list)
plt.title('Episodic Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Try validation set, {vary test size and random state: done}
