import tensorflow as tf
import pprint as pp
from SRCNN import SRCNN

def setupFlags():
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 3, "[15000]")
    flags.DEFINE_integer("batch_size", 128, "[128]")
    flags.DEFINE_integer("image_size", 33, "[33]")
    flags.DEFINE_integer("label_size", 21, "[21]")
    flags.DEFINE_float("learning_rate", 1e-4, "[1e-4]")
    flags.DEFINE_integer("c_dim", 1, "[1]")
    flags.DEFINE_boolean("is_greyscale", (flags.FLAGS.c_dim == 1), "[-]")
    flags.DEFINE_integer("scale", 3, "[3]")
    flags.DEFINE_integer("stride", 21, "[14]")
    flags.DEFINE_string("checkpoint_dir", "checkpoint", "[checkpoint]")
    flags.DEFINE_string("sample_dir", "sample", "[sample]")
    flags.DEFINE_boolean("is_train", False, "[True]")
    flags.DEFINE_string("train_dir", "Train", "[Train]")
    flags.DEFINE_string("test_dir", "Test", "[Test]")
    flags.DEFINE_string("test_set", "Set5", "[Set5]")

    # Model descriptors
    flags.DEFINE_integer("f1", 9, "[9]")
    flags.DEFINE_integer("f2", 1, "[1]")
    flags.DEFINE_integer("f3", 5, "[5]")

    flags.DEFINE_integer("n1", 64, "[64]")
    flags.DEFINE_integer("n2", 32, "[32]")

    return flags.FLAGS

def main(argv):
    # Set up flags
    FLAGS = setupFlags()
    printer = pp.PrettyPrinter()
    printer.pprint(FLAGS.__flags)

    with tf.Session() as sess:
        # Initialize SRCNN Model - Creates CNN layers, loss function and other model related variables
        resBooster_SRCNN = SRCNN(sess, FLAGS)

        # Load Model
        print(" [*] Loading model...")
        model_loaded = resBooster_SRCNN.load_model()
        if(model_loaded):
            print(" [*] Model successfully loaded.")
        else:
            print(" [*] Error loading model.")

        # Train model / Test
        if(FLAGS.is_train):
            resBooster_SRCNN.train_model()
        else:
            resBooster_SRCNN.test_model()

if __name__ == '__main__':
  tf.app.run()


