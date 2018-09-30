# r_tensorflow_gpu_speed
A test of CPU vs. GPU speed for R + TensorFlow.

This test uses a recurrent neural network (RNN) with random data as inputs.

The files are:

1. "sample_install_commands.txt" gives one example of creating a Conda environment, with TensorFlow, that will be called by R.
2. "r_gpu_speed.r" is the main benchmarking routine.
3. "sample_output.txt" shows what the console output looks like.
4. "sample_results.png" is a chart that is created automatically if "screen_display <- FALSE" in the main program.

This test chows that GPUs can be much faster that CPUs even for sequential analysis, which are not even the tasks at which GPUs excel.

Moreover, the cuDNN library provides extra GPU speed over the standard TensorFlow RNN libraries.

![CPU < GPU < GPU+cuDNN](https://github.com/hergott/r_tensorflow_gpu_speed/sample_results.png)
