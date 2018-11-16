import tensorflow as tf
from SRCNN_utils import *
import time
xrange = range

class SRCNN(object):
    def __init__(self, tf_sess, flags):
        self.FLAGS = flags
        self.TF_SESS = tf_sess
        self.build_model_arguments()
        self.build_model()
        self.build_loss_function()

    ################################ MODEL INITIALIZATION FUNCTIONS ################################

    def build_model_arguments(self):
        self.IMAGES = tf.placeholder(tf.float32, [None, self.FLAGS.image_size, self.FLAGS.image_size, self.FLAGS.c_dim], name='images')
        self.LABELS = tf.placeholder(tf.float32, [None, self.FLAGS.label_size, self.FLAGS.label_size, self.FLAGS.c_dim], name='labels')

        self.WEIGHTS = {
            'w1': tf.Variable(tf.random_normal([self.FLAGS.f1, self.FLAGS.f1, self.FLAGS.c_dim, self.FLAGS.n1], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([self.FLAGS.f2, self.FLAGS.f2, self.FLAGS.n1, self.FLAGS.n2], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([self.FLAGS.f3, self.FLAGS.f3, self.FLAGS.n2, self.FLAGS.c_dim], stddev=1e-3), name='w3')
        }

        self.BIASES = {
            'b1': tf.Variable(tf.zeros([self.FLAGS.n1]), name='b1'),
            'b2': tf.Variable(tf.zeros([self.FLAGS.n2]), name='b2'),
            'b3': tf.Variable(tf.zeros([self.FLAGS.c_dim]), name='b3')
        }

    def build_model(self):
        self.LAYER1 = tf.nn.relu(
            tf.nn.conv2d(self.IMAGES, self.WEIGHTS['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.BIASES['b1'])
        self.LAYER2 = tf.nn.relu(
            tf.nn.conv2d(self.LAYER1, self.WEIGHTS['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.BIASES['b2'])
        self.LAYER3 = tf.nn.conv2d(self.LAYER2, self.WEIGHTS['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.BIASES['b3']

        self.SAVER = tf.train.Saver()

    def build_loss_function(self):
        self.LOSS = tf.reduce_mean(tf.square(self.LABELS - self.LAYER3))

        # Stochastic gradient descent with backprop
        self.TRAINER = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate).minimize(self.LOSS)

    def load_model(self):
        tf.global_variables_initializer()
        checkpoint_dir = os.path.join(self.FLAGS.checkpoint_dir, "srcnn_" + str(self.FLAGS.label_size))
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        self.TRAINING_CYCLES = 0  # Corresponds to the number of training batches that have passed through the cnn
        checkpoint_loaded = False
        if checkpoint and checkpoint.model_checkpoint_path:
            self.SAVER.restore(self.TF_SESS, os.path.join(checkpoint_dir, os.path.basename(checkpoint.model_checkpoint_path)))
            checkpoint_loaded = True
            # TODO: Change TRAINING_CYCLE based on loaded model
        return checkpoint_loaded

    ################################ TRAINING FUNCTIONS ################################

    def train_model(self):
        # Preprocess input and store training data in h5 file
        numPatchesX, numPatchesY = setupInput(self.FLAGS) # If training, numPatchesX = numPatchesY = None

        # Load training data from h5 file
        training_data_path = os.path.join('./{}'.format(self.FLAGS.checkpoint_dir), "train.h5")
        training_data, training_labels = readInputsAndLabelsFromH5(training_data_path)

        # Start Training
        self.START_TIME = time.time()

        for self.TRAINING_EPOCH in xrange(self.FLAGS.epoch):
            for i in range(0, len(training_data), self.FLAGS.batch_size):
                self.TRAINING_CYCLES += 1
                cur_image_patches = training_data[i:i+self.FLAGS.batch_size]
                cur_image_patchlabels = training_labels[i:i+self.FLAGS.batch_size]
                _, self.MODEL_ERROR = self.TF_SESS.run([self.TRAINER, self.LOSS], feed_dict={self.IMAGES: cur_image_patches, self.LABELS: cur_image_patchlabels})
                self.periodicPrint()
                self.periodicSave()

    def periodicPrint(self):
        if(self.TRAINING_CYCLES%10 == 0):
            epoch = ("%2d"% self.TRAINING_EPOCH)
            training_cycles = ("%2d"% self.TRAINING_CYCLES)
            run_time = time.time() - self.START_TIME
            run_time = ("%4.4f"% run_time)
            loss = (".8f"% self.MODEL_ERROR)
            print(" [*] Epoch: " + epoch + " Training Cycles: " + training_cycles + " Run Time: " + run_time + " Error/Loss: " + loss)

    def periodicSave(self):
        if(self.TRAINING_CYCLES%500 == 0):
            self.save_model()

    def save_model(self):
        checkpoint_dir = os.path.join(self.FLAGS.checkpoint_dir, "srcnn_" + str(self.FLAGS.label_size))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.SAVER.save(self.TF_SESS, os.path.join(checkpoint_dir, "SRCNN.model"), global_step=self.TRAINING_CYCLES)

    ################################ TESTING FUNCTIONS ################################

    def test_model(self):
        # Preprocess input and store training data in h5 file
        numPatchesX, numPatchesY = setupInput(self.FLAGS)

        # Load training data from h5 file
        testing_data_path = os.path.join('./{}'.format(self.FLAGS.checkpoint_dir), "test.h5")
        testing_data, testing_labels = readInputsAndLabelsFromH5(testing_data_path)

        # Get resolution boosted patches
        resolution_boosted_patches = self.LAYER3.eval({self.IMAGES: testing_data, self.LABELS: testing_labels})

        # Merge res boosted patches to form image
        resolution_boosted_image = (self.merge(resolution_boosted_patches, numPatchesX, numPatchesY)).squeeze()

        # Save image
        # scipy.misc.imsave(os.path.join(os.path.join(os.getcwd(), self.FLAGS.sample_dir), "test_image.png)"), resolution_boosted_image)
        showImage(resolution_boosted_image, "sr4")

    def merge(self, patches, numPatchesX, numPatchesY):
        merged_image = np.zeros((patches.shape[1] * numPatchesX, patches.shape[2] * numPatchesY, 1))

        for patchNum, patch in enumerate(patches):
            i = patchNum % numPatchesY
            j = patchNum // numPatchesY
            merged_image[j * patches.shape[1]:j * patches.shape[1] + patches.shape[1], i * patches.shape[2]:i * patches.shape[2] + patches.shape[2], :] = patch
        return merged_image