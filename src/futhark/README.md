
## Experimental results

Running with the following configuration:

    size of training data: 50.000 images and associated digits
    size of test data: 10.000 images and associated digits
    epochs: 10
    mini_batch_size: 400
    eta: 5.0

Execution outputs the prediction rate for the test images after the
network has been taught with the training data.

All experiments are made on a Mac Book Pro (Intel Core i7, 2,7GHz CPU,
AMD Radeon Pro 460 GPU):

    bash-3.2$ uname -a
    Darwin Martins-MBP 16.4.0 Darwin Kernel Version 16.4.0: Thu Dec 22 22:53:21 PST 2016; root:xnu-3789.41.3~3/RELEASE_X86_64 x86_64

### AMD GPU execution (f32)

    real  0m47.067s
    user  0m38.967s
    sys   0m1.106s
    Image digit prediction percentage:
    78.820000f32
    Execution time (ex. data load):
    6898538

### AMD GPU execution (f64)

    real  0m49.822s
    user  0m38.033s
    sys   0m1.188s
    Image digit prediction percentage:
    79.010000f64
    Execution time (ex. data load):
    10396590

### CPU execution (f32)

    real  1m28.255s
    user  1m27.765s
    sys   0m0.554s
    Image digit prediction percentage:
    82.129997f32
    Execution time (ex data load):
    49499496

### CPU execution (f64)

    real  1m38.318s
    user  1m37.540s
    sys   0m0.843s
    Image digit prediction percentage:
    82.130000f64
    Execution time (ex data load):
    61183626
