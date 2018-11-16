import os
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py



###################################### IMAGE PRE-PROCESSING FUNCTIONS ######################################
def setupInput(FLAGS):
    file_list = getFileNames(FLAGS)

    # len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    # (sub_input_sequence[0]).shape : (33, 33, 1)
    sub_input_sequence = [] # TODO initialize as nparray
    sub_label_sequence = []
    numPatchesX = None
    numPatchesY = None

    # Generate patches for input image, and the corresponding label patch for each
    if FLAGS.is_train:
        for f in file_list:
            input_patches, label_patches = getPatches(FLAGS, f)
            sub_input_sequence.extend(input_patches)
            sub_label_sequence.extend(label_patches)
    else:
        TEST_IMAGE = file_list[0] # CHANGE TO TEST A DIFFERENT IMAGE
        input_patches, label_patches = getPatches(FLAGS, TEST_IMAGE)
        sub_input_sequence.extend(input_patches)
        sub_label_sequence.extend(label_patches)

        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        inputIm, label = preProcess(FLAGS, TEST_IMAGE)
        numPatchesX = math.ceil((inputIm.shape[0] - FLAGS.image_size + 1)/FLAGS.stride)
        numPatchesY = math.ceil((inputIm.shape[1] - FLAGS.image_size + 1)/FLAGS.stride)

    saveInputsAndLabelsToH5(FLAGS, np.asarray(sub_input_sequence), np.asarray(sub_label_sequence))
    return numPatchesX, numPatchesY # Returns None, None if Training


def getFileNames(FLAGS):
    if (FLAGS.is_train):
        dataDirectory = os.path.join(os.getcwd(), FLAGS.train_dir)
    else:
        dataDirectory = os.path.join(os.sep, (os.path.join(os.getcwd(), FLAGS.test_dir)), FLAGS.test_set)
    return glob.glob(os.path.join(dataDirectory, "*.bmp"))


def preProcess(FLAGS, fn, scale=3): #Scale affects the kind of blurring
    # Load Image
    im = scipy.misc.imread(fn, flatten=FLAGS.is_greyscale, mode='YCbCr').astype(np.float)

    # Crop Image Edges
    if(im.shape == 3):
        label = im[0:(im.shape[0] - np.mod(im.shape[0], scale)), 0:(im.shape[1] - np.mod(im.shape[1], scale)), :]
    else:
        label = im[0:(im.shape[0] - np.mod(im.shape[0], scale)), 0:(im.shape[1] - np.mod(im.shape[1], scale))]

    # Normalize label
    label = label/255

    # Generate low-res input (Y), and ground-truth high-res label (X)
    input = scipy.ndimage.interpolation.zoom(label, (1. / scale), prefilter=False)
    input = scipy.ndimage.interpolation.zoom(input, (scale / 1.), prefilter=False)
    showImage(input, "input4")
    showImage(label, "label4")

    return input, label

def getPatches(FLAGS, f):
    inputIm, label = preProcess(FLAGS, f)
    showImage(inputIm)
    showImage(label)

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(FLAGS.image_size - FLAGS.label_size) / 2  # 6
    for x in range(0, inputIm.shape[0] - FLAGS.image_size + 1, FLAGS.stride):
        for y in range(0, inputIm.shape[1] - FLAGS.image_size + 1, FLAGS.stride):
            sub_input = inputIm[x:x + FLAGS.image_size, y:y + FLAGS.image_size]  # [33 x 33]
            sub_label = label[x + int(padding):x + int(padding) + FLAGS.label_size,
                        y + int(padding):y + int(padding) + FLAGS.label_size]  # [21 x 21]

            # Make channel value
            sub_input = sub_input.reshape([FLAGS.image_size, FLAGS.image_size, 1])
            sub_label = sub_label.reshape([FLAGS.label_size, FLAGS.label_size, 1])

            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)

    return sub_input_sequence, sub_label_sequence


def saveInputsAndLabelsToH5(FLAGS, inputIm, label):
    # Make input data as h5 file format. Depending on 'is_train' (flag value), savepath would be changed.
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=inputIm)
        hf.create_dataset('label', data=label)


###################################### OTHER SRCNN HELPER FUNCTIONS ######################################

def readInputsAndLabelsFromH5(file_path):
    with h5py.File(file_path, 'r') as hf:
        return np.array(hf.get('data')), np.array(hf.get('label'))

def showImage(im, fn="unknown"):
    print(fn, ": ", im.shape)
    plt.imshow(im, cmap="gray")
    plt.savefig("temp/"+fn+".png")
    plt.show()
