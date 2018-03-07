"""mnist_to_fut_format
~~~~~~~~~~~~~~~~~~~

A library to load the MNIST image data and convert it into a format
compatible with Futhark.
"""

#### Libraries
# Standard library
import _pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def out_flat_arr(s, a):
    f = open(s,'w')
    f.write('[')
    l = len(a)
    for i in range(0,l-1):
        f.write(str(a[i]) + ",")
    if (l > i):
        f.write(str(a[i]))
    f.write(']')
    f.close

def out_fut_data(s,n,d):
    inputs = d[0]
    results = d[1]
    out_flat_arr('mnist_'+s+'_input.fut.txt',np.reshape(inputs[:n], n*28*28))
    out_flat_arr('mnist_'+s+'_results.fut.txt',results[:n])

def to_fut_format():
    """Output files that can be loaded by a Futhark-generated
    executable. The files include
        1) a file with training inputs (size 50.000 x 28 x 28 flat float array)
        2) a file with training outputs (size 50.000 flat int array)
        3) a file with validation inputs (size 10.000 x 28 x 28 flat float array)
        4) a file with validation outputs (size 10.000 flat int array)
        5) a file with test inputs (size 10.000 x 28 x 28 flat float array)
        6) a file with test outputs (size 10.000 flat int array)
    Each output file contains a flat float array with an entries being integers
    (digits) corresponding to the input image."""
    tr_d, va_d, te_d = load_data()
    out_fut_data('training',50000,tr_d)
    out_fut_data('validation',10000,va_d)
    out_fut_data('test',10000,te_d)
    out_fut_data('training_small',5000,tr_d)
    out_fut_data('validation_small',1000,va_d)
    out_fut_data('test_small',1000,te_d)

to_fut_format()
