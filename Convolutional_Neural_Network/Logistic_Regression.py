import sys
import numpy
from utils import*

class Logistic_Regression():
    
    def __init__(self, In, Out):

        self.In  = In
        self.Out = Out

        print "In"
        print In
        print "Out"
        print Out

        uniform_boundary = 1.0 / In
        self.W = numpy.array(numpy.random.uniform(
                                low = -uniform_boundary,
                                high = uniform_boundary,
                                size = (self.Out, self.In)
                                )
                            )

        self.b = numpy.zeros(self.Out)

    def train(self, Input, T, learning_rate):

        minibatch_size = len(Input)
        delta = numpy.zeros((minibatch_size, self.Out))
        grad_W = numpy.zeros((self.Out, self.In))
        grad_b = numpy.zeros((self.Out))
        predicted_y = numpy.zeros((self.Out))

        for batch_roop in xrange(minibatch_size):
            predicted_y = self.output(Input[batch_roop])
            print "batch_roop"
            print batch_roop
            print "predicted"
            print predicted_y

            for out_roop in xrange(self.Out):
                delta[batch_roop][out_roop] = predicted_y[out_roop] - T[batch_roop][out_roop]

                for in_roop in xrange(self.In):
                    grad_W[out_roop][in_roop] += delta[batch_roop][out_roop] * Input[batch_roop][in_roop]
                
                grad_b[out_roop] += delta[batch_roop][out_roop]
                
        for out_roop in xrange(self.Out):
            for in_roop in xrange(self.In):
                self.W[out_roop][in_roop] -= learning_rate * grad_W[out_roop][in_roop] / minibatch_size
            
            self.b[out_roop] -= learning_rate * grad_b[out_roop] / minibatch_size

        self.delta = delta

    def output(self, Input):

        output_preatctivation = numpy.zeros(self.Out)

        for out_roop in xrange(self.Out):
            for in_roop in xrange(self.In):
                output_preatctivation[out_roop] += self.W[out_roop][in_roop] * Input[in_roop]
        
            output_preatctivation[out_roop] += self.b[out_roop]
        
        return self.soft_max(output_preatctivation)

    def predict(self, x):

        minibatch_size = len(x)
        label = numpy.zeros((minibatch_size, self.Out))
        expected = numpy.zeros((minibatch_size, self.Out))

        for batch_roop in xrange(minibatch_size):
            expected[batch_roop] = self.output(x[batch_roop])

        arg_max = numpy.zeros(minibatch_size)

        for batch_roop in xrange(minibatch_size):
            Max = 0.0
            for out_roop in xrange(self.Out):
                if(Max < expected[batch_roop][out_roop]):
                    Max = expected[batch_roop][out_roop]
                    arg_max[batch_roop] = out_roop
        
        for batch_roop in xrange(minibatch_size):
            for out_roop in xrange(self.Out):
                if(out_roop == arg_max[batch_roop]):
                    label[batch_roop][out_roop] = 1
        
        print label
        return label

    def soft_max(self, x):

        softmax_y = numpy.zeros(self.Out)

        Max = 0.0
        Sum = 0.0

        for i in xrange(self.Out):
            if Max < x[i]:
                Max = x[i]
        
        for i in xrange(self.Out):
            softmax_y[i] += numpy.exp(x[i] - Max)
            Sum += softmax_y[i]
    
        for i in xrange(self.Out):
            softmax_y[i] /= Sum
        
        return softmax_y