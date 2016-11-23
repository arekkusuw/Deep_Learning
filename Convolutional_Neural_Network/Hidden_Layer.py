import sys
import numpy
from utils import*

class Hidden_Layer():
    
    def __init__(self, In, Out, drop_out_possibility):

        self.In  = In
        self.Out = Out
        self.drop_out_possibility = drop_out_possibility

        print "In"
        print In
        print "Out"
        print Out
        print "drop_out_possibility"
        print drop_out_possibility
        print

        uniform_boundary = 1.0 / In
        self.W = numpy.array(numpy.random.uniform(
                            low = -uniform_boundary,
                            high = uniform_boundary,
                            size = (Out, In)
                            )
                        )
        
        self.b = numpy.zeros(Out)
        
    def forward(self, Input):
        self.Input = Input
        return self.output(self.Input)

    def backward(self, prev_layer, learning_rate):

        minibatch_size = len(prev_layer.delta)
        delta = numpy.zeros((minibatch_size, self.Out))
        grad_W = numpy.zeros((self.Out, self.In))
        grad_b = numpy.zeros((self.Out))

        for batch_roop in xrange(minibatch_size):
            for prev_layer_in_roop in xrange(self.Out):
            #prev_layer_in means self layer out
                for prev_layer_out_roop in xrange(prev_layer.Out):
                    
                    delta[batch_roop][prev_layer_in_roop] += prev_layer.W[prev_layer_out_roop][prev_layer_in_roop] * prev_layer.delta[batch_roop][prev_layer_out_roop]
                    delta[batch_roop][prev_layer_in_roop] *= dReLU(self.Output[batch_roop][prev_layer_in_roop])
                    
                    delta[batch_roop] *= self.dropout_masks[batch_roop]
                    
                    for in_roop in xrange(int(self.In)):
                        grad_W[prev_layer_in_roop][in_roop] += delta[batch_roop][prev_layer_in_roop] * self.Input[batch_roop][in_roop]
                        grad_b[prev_layer_in_roop] += delta[batch_roop][prev_layer_in_roop]
        
        for out_roop in xrange(int(self.Out)):
            for in_roop in xrange(int(self.In)):
                self.W[out_roop][in_roop] -= learning_rate * grad_W[out_roop][in_roop] / minibatch_size
                self.b[out_roop] -= learning_rate * grad_b[out_roop] / minibatch_size
        
        self.delta = delta

    def output(self, Input):

        minibatch_size = len(Input)
        Output = numpy.zeros( (minibatch_size, self.Out) )
        dropout_masks = numpy.zeros( (minibatch_size, self.Out) )
        
        for batch_roop in xrange(minibatch_size):
            for out_roop in xrange(self.Out):
                pre_activation = 0.0
                
                for in_roop in xrange(int(self.In)):
                    pre_activation += self.W[out_roop][in_roop] * Input[batch_roop][in_roop]
            
            pre_activation += self.b[out_roop]
            Output[batch_roop][out_roop] += ReLU(pre_activation)

            dropout_masks[batch_roop], Output[batch_roop] = self.dropout(Output[batch_roop], self.drop_out_possibility)
            
            
        self.dropout_masks = dropout_masks
        self.Output = Output
        return Output
        
    def dropout(self, Input, p):
        mask = numpy.random.binomial(size = len(Input), n = 1, p = 1 - p)
        Input *= mask
        return [mask, Input]