# TensorFlow package for R:
# https://cran.r-project.org/web/packages/tensorflow/index.html
#
# Reticulate package used to set Python and Numpy random seeds:
# https://cran.r-project.org/web/packages/reticulate/index.html
#
#install.packages("tensorflow")
#install.packages("reticulate")
#install.packages("RColorBrewer")
 

# This way of using TensorFlow with R utilizes an Anaconda environment that has all the
# appropriate Python and TensorFlow dependencies installed.
#
library(tensorflow)
use_condaenv("r-tensorflow", required=TRUE)


# Test R + TensorFlow installation.
#
test_tensorflow_install <- function(){
    sess <- tf$Session()
    hello <- tf$constant('Hello, TensorFlow!')
    out <- sess$run(hello)

    print(hello)
    print(out)
}


# Set random seeds.
#
# For more detailed discussion of R+Tensorflow seeds, 
# see: https://github.com/rstudio/tensorflow/blob/master/R/seed.R
#
set_random_seeds <- function(){
    set.seed(123)
    tf$set_random_seed(1234)
    reticulate::py_set_seed(12345)
}


# Make recurrent neural network (RNN) computational graph for TensorFlow.
#
create_rnn_graph <- function(x, n_units, cudnn=FALSE){
    if (cudnn) {
        cudnn_gru_layer <- tf$contrib$cudnn_rnn$CudnnGRU(num_layers=1L, num_units=n_units, dtype=tf$float32)
        rnn_outputs <- cudnn_gru_layer(x)        
    } else {
        x_split <- tf$unstack(x)
        rnn_cell <- tf$contrib$rnn$GRUCell(num_units=n_units, dtype=tf$float32)
        rnn_outputs <- tf$contrib$rnn$static_rnn(rnn_cell, x_split, dtype=tf$float32)
    }
    rnn_state <- rnn_outputs[2]

    rnn_state_squeeze <- tf$squeeze(rnn_state)

    logits <- tf$layers$dense(inputs=rnn_state_squeeze, units=1L, activation=NULL)

    return(logits)
}


# Main benchmarking routine.
#
tf_speed_test <- function(){

    # Dimensions: larger RNNs should increase relative advantage of GPU and cuDNN.
    sequence_dim <- 128L
    sequence_len <- 64L
    n_samples <- 1024L
    n_epochs <- 50
    n_rnn_units <- 512L

    # Matrices to hold results.
    epoch_times <- matrix(0, n_epochs, 3)
    epoch_losses <- matrix(0, n_epochs, 3)
    
    # Three different test runs.
    labels <- c("CPU", "GPU + CUDA", "GPU + CUDA + cuDNN")
    use_GPU <- c(FALSE, TRUE, TRUE)
    use_cudNN <- c(FALSE, FALSE, TRUE)

    for (config in c(2, 3, 1)) {
        cat("\n", "Starting session for ", labels[config], ".", "\n", sep = "")
        
        tf$reset_default_graph()
        
        set_random_seeds()
        
        # Use CPU or GPU+CUDA (here, 1 GPU).
        sess_config <- list()
	cat("Use GPU:",use_GPU[config])
        if (use_GPU[config]){
	    print("USING GPU")
            Sys.setenv(CUDA_VISIBLE_DEVICES = "0")
            sess_config$device_count <- list(GPU = 1L, CPU = 1L)
        } else {
	    print("NOT USING GPU")
            Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
            sess_config$device_count <- list(GPU = 0L, CPU = 1L)
        }

        session_conf <- do.call(tf$ConfigProto, sess_config)
        
        # Random data for RNN.
        x <- tf$Variable(tf$random_uniform(c(sequence_len, n_samples, sequence_dim), -1.0, 1.0))
        y <- tf$round(tf$Variable(tf$random_uniform(c(n_samples, 1L), 0.0, 1.0)))
        
        rnn_logits <- create_rnn_graph(x, n_rnn_units, cudnn=use_cudNN[config])

        rnn_loss <- tf$nn$sigmoid_cross_entropy_with_logits(labels=y, logits=rnn_logits)

	mean_loss <- tf$reduce_mean(rnn_loss)
        
        optimizer <- tf$train$AdamOptimizer()
        train <- optimizer$minimize(rnn_loss)

        sess <- tf$Session(graph = tf$get_default_graph(), config = session_conf)
        sess$run(tf$global_variables_initializer())

        for (epoch in 1:n_epochs) {
            p_time <- proc.time()
            
            step_run <- sess$run(train)
            
            loss <- mean_loss$eval(session=sess)
            
            epoch_losses[epoch, config] <- loss
            
            e_time <- proc.time()["elapsed"] - p_time["elapsed"]
            
            epoch_times[epoch, config] <- e_time
            
            cat("session: ", labels[config], "      epoch: ", epoch, "      time: ", 
                 e_time, "      loss: ", round(loss, digits=5), "\n", sep="")
        } 

        sess$close()
    }
    
    # Print results.
    cat("\nTimes:\n")
    print(epoch_times)
    cat("\nLosses:\n")
    print(epoch_losses)
    
    # Chart losses.
    #
    # Please note that cuDNN uses a slightly different RNN formulation,
    # so the raw numbers aren't directly comparable.
    
    # Set output to screen or PNG file.
    screen_display <- FALSE
    if (screen_display) {
        l_width <- 6
    } else {
        png(filename="results.png", width = 1080, height = 1080, units = "px", pointsize = 22, bg = "white", type = "cairo")#c("calibri", "cairo", "cairo-png", "Xlib", "quartz"))
        l_width <- 10
    }

    par(mfrow=c(2,1), mgp=c(1.9,0.6,0),mar=c(4,3,3,1)+0.1, font.lab=2 )
    library(RColorBrewer)
    # display.brewer.all(3)
    
    # Chart epoch times.
    col <- brewer.pal(3,"Dark2")
    matplot(epoch_times, type = "l", xlab="Epoch #", ylab="Time (seconds)", col=col, lwd=c(l_width, l_width, l_width))
    legend("topright", inset=0.01, legend=labels, col=col, pch=c(15:17), bg= ("white"), cex = 0.85)
    title("Time per epoch", cex=1.1)
    
    # Chart losses.
    col <- brewer.pal(3,"Set1")
    matplot(epoch_losses, type = "l", xlab="Epoch #", ylab="Loss", col=col, lwd=c(l_width, l_width, l_width))
    legend("topright", inset=0.01, legend=labels, col=col, pch=c(15:17), bg= ("white"), cex = 0.85)
    title("Losses (note: cuDNN raw numbers not directly comparable)", cex=1.1)
    
    if (!screen_display) {
        dev.off()
    }
}


# Run functions to (1) test TensorFlow install, and (2) main benchmarking routine.
#
#test_tensorflow_install()
tf_speed_test()





