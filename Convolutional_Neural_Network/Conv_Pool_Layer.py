import sys
import numpy
from utils import*

class Conv_Pool_Layer(object):
    
    def __init__(self, image_size, channel, number_of_kernels, kernel_size, pool_size, convolved_size, pooled_size, minibatch_size):
         
        self.image_size        = image_size
        self.channel           = channel
        self.number_of_kernels = number_of_kernels
        self.kernel_size       = kernel_size
        self.pool_size         = pool_size
        self.convolved_size    = convolved_size
        self.pooled_size       = pooled_size

        print "image_size"
        print image_size
        print "channel"
        print channel
        print "number_of_kernels"
        print number_of_kernels
        print "kernel_size"
        print kernel_size
        print "pool_size"
        print pool_size
        print "convolved_size"
        print convolved_size
        print "pooled_size"
        print pooled_size
        print

        conv_in = channel * kernel_size[0] * kernel_size[1]
        conv_out = number_of_kernels * kernel_size[0] * kernel_size[1] / (pool_size[0] * pool_size[1])
        uniform_boundary = numpy.sqrt(6.0 / (conv_in + conv_out))
     
        self.W = numpy.array(numpy.random.uniform(
                                low = -uniform_boundary,
                                high = uniform_boundary,
                                size = (number_of_kernels, channel, kernel_size[0], kernel_size[1])
                            )
                        )
 
        self.b = numpy.zeros(number_of_kernels)

        self.dconved_delta = numpy.zeros((minibatch_size, self.channel, self.image_size[0], self.image_size[1]))
 
    def forward(self, input):
        
        z = self.convolve(input)
        return self.max_pooling(z)
 
    def backward(self, prev_layer_delta, learning_rate):
        
        delta = self.dmax_pooling(prev_layer_delta, self.activated_input, learning_rate)
        self.dconvolve(delta, learning_rate)
 
    def convolve(self, input):
        
        minibatch_size = len(input)
        pre_activated_input = numpy.zeros((minibatch_size, self.number_of_kernels, self.convolved_size[0], self.convolved_size[1]))
        activated_input = numpy.zeros((minibatch_size, self.number_of_kernels, self.convolved_size[0], self.convolved_size[1]))
 
        for batch_roop in xrange(minibatch_size):
            for num_ker_roop in xrange(self.number_of_kernels):
                for conved_col_roop in xrange(int(self.convolved_size[0])):
                    for conved_row_roop in xrange(int(self.convolved_size[1])):
                 
                        for channel_roop in xrange(self.channel):
                            for ker_col_roop in xrange(self.kernel_size[0]):
                                for ker_row_roop in xrange(self.kernel_size[1]):
                             
                                    pre_activated_input[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop] += self.W[num_ker_roop][channel_roop][ker_col_roop][ker_row_roop] * input[batch_roop][channel_roop][conved_col_roop + ker_col_roop][conved_row_roop + ker_row_roop]
                         
                        pre_activated_input[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop] = pre_activated_input[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop] + self.b[num_ker_roop]
                        activated_input[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop]     = ReLU(pre_activated_input[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop])
                         
        self.input = input
        self.pre_activated_input = pre_activated_input
        self.activated_input = activated_input
         
        return activated_input
 
    def dconvolve(self, prev_layer_delta, learning_rate):
        
        minibatch_size = len(prev_layer_delta)
        grad_W = numpy.zeros( (self.number_of_kernels, self.channel, self.kernel_size[0], self.kernel_size[1]) )
        grad_b = numpy.zeros( self.number_of_kernels )
 
        for batch_roop in xrange(minibatch_size):
            for num_ker_roop in xrange(self.number_of_kernels):
                for conved_col_roop in xrange(int(self.convolved_size[0])):
                    for conved_row_roop in xrange(int(self.convolved_size[1])):
 
                        d = prev_layer_delta[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop] * dReLU(self.pre_activated_input[batch_roop][num_ker_roop][conved_col_roop][conved_row_roop])
                        grad_b[num_ker_roop] += d
 
                    for channel_roop in xrange(self.channel):
                        for ker_col_roop in xrange(self.kernel_size[0]):
                            for ker_row_roop in xrange(self.kernel_size[1]):
 
                                grad_W[num_ker_roop][channel_roop][ker_col_roop][ker_row_roop] = d * self.input[batch_roop][channel_roop][conved_col_roop + ker_col_roop][conved_row_roop + ker_row_roop]
         
        for num_ker_roop in xrange(self.number_of_kernels):
            self.b[num_ker_roop] -= learning_rate * grad_b[num_ker_roop] / minibatch_size
            for channel_roop in xrange(self.channel):
                for ker_col_roop in xrange(self.kernel_size[0]):
                    for ker_row_roop in xrange(self.kernel_size[1]):
                        self.W[num_ker_roop][channel_roop][ker_col_roop][ker_row_roop] -= learning_rate * grad_W[num_ker_roop][channel_roop][ker_col_roop][ker_row_roop] / minibatch_size
       
 
        for batch_roop in xrange(minibatch_size):
            for channel_roop in xrange(self.channel):
                for image_col_roop in xrange(int(self.image_size[0])):
                    for image_row_roop in xrange(int(self.image_size[1])):
                        for num_ker_roop in xrange(self.number_of_kernels):
                            for ker_col_roop in xrange(self.kernel_size[0]):
                                for ker_row_roop in xrange(self.kernel_size[1]):
 
                                    d =0.0
                                 
                                    if (image_col_roop - (self.kernel_size[0] - 1) - ker_col_roop >= 0) and (image_row_roop - (self.kernel_size[1] - 1) - ker_row_roop >= 0):
                                     
                                        d = prev_layer_delta[batch_roop][num_ker_roop][image_col_roop - (self.kernel_size[0] - 1) - ker_col_roop][image_row_roop - (self.kernel_size[1] - 1) - ker_row_roop] *  dReLU(self.pre_activated_input[batch_roop][num_ker_roop][image_col_roop - (self.kernel_size[0] -1) - ker_col_roop][image_row_roop - (self.kernel_size[1] - 1) - ker_row_roop]) * self.W[num_ker_roop][channel_roop][ker_col_roop][ker_row_roop]
                                     
                                    self.dconved_delta[batch_roop][channel_roop][image_col_roop][image_row_roop] += d
       
    def max_pooling(self, input):
        
        minibatch_size = len(input)
        pooled_input = numpy.zeros( (minibatch_size, self.number_of_kernels, self.pooled_size[0], self.pooled_size[1]) )
         
        for batch_roop in xrange(minibatch_size):
            for num_ker_roop in xrange(self.number_of_kernels):
                for pooled_col_roop in xrange(int(self.pooled_size[0])):
                    for pooled_row_roop in xrange(int(self.pooled_size[1])):
 
                        max = 0.0
 
                        for pool_col_roop in xrange(self.pool_size[0]):
                            for pool_row_roop in xrange(self.pool_size[1]): 
 
                                if (pool_col_roop == 0) and (pool_row_roop == 0):
                                    max = input[batch_roop][num_ker_roop][self.pool_size[0] * pooled_col_roop][self.pool_size[1] * pooled_row_roop]
                                    next
                         
                                if (max < input[batch_roop][num_ker_roop][self.pool_size[0] * pooled_col_roop + pool_col_roop][self.pool_size[1] * pooled_row_roop + pool_row_roop]):
                                    max = input[batch_roop][num_ker_roop][self.pool_size[0] * pooled_col_roop + pool_col_roop][self.pool_size[1] * pooled_row_roop + pool_row_roop]
                    
                        pooled_input[batch_roop][num_ker_roop][pooled_col_roop][pooled_row_roop] = max
 
        self.pooled_input = pooled_input
        return pooled_input
 
 
    def dmax_pooling(self, prev_layer_delta, layer_input, learning_rate):
    
        minibatch_size = len(prev_layer_delta)
   
        delta = numpy.zeros((minibatch_size, self.number_of_kernels, self.convolved_size[0], self.convolved_size[1]))
 
        for batch_roop in xrange(minibatch_size):
            for num_ker_roop in xrange(self.number_of_kernels):
                for pooled_col_roop in xrange(int(self.pooled_size[0])):
                    for pooled_row_roop in xrange(int(self.pooled_size[1])):
                        for pool_col_roop in xrange(self.pool_size[0]): 
                            for pool_row_roop in xrange(self.pool_size[1]):
 
                                d = 0.0
 
                                if self.pooled_input[batch_roop][num_ker_roop][pooled_col_roop][pooled_row_roop] == layer_input[batch_roop][num_ker_roop][self.pool_size[0] * pooled_col_roop + pool_col_roop][self.pool_size[1] * pooled_row_roop + pool_row_roop]:
                                     
                                    d = prev_layer_delta[batch_roop][num_ker_roop][pooled_col_roop][pooled_row_roop]
                                 
                                delta[batch_roop][num_ker_roop][self.pool_size[0] * pooled_col_roop + pool_col_roop][self.pool_size[1] * pooled_row_roop + pool_row_roop] = d
             
        return delta