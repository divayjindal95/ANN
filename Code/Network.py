import numpy as np
import pandas as pd

class Network:
    def __init__(self,sizes,epoches,batch_size,eta):
        self.sizes=sizes
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.biases=[np.random.randn(y,1) for y in sizes[1:] ]
        self.epoches=epoches
        self.batch_size=batch_size
        self.eta=eta

    def stochastic_gradient_descent(self,train_data,test_data):
        
        for e in xrange(self.epoches):
            
            np.random.shuffle(train_data)

            mini_batches=[ train_data[i:i+self.batch_size] for i in xrange(0,len(train_data),self.batch_size) ]
            for mini_batch in mini_batches:
                delta_w=[np.zeros(np.shape(y)) for y in self.weights]
                delta_b=[np.zeros(np.shape(y)) for y in self.biases]
                #print delta_b
                for single_sample in mini_batch:
                    #print len(single_sample[0])
                    activations,Z=self.feed_forward(single_sample[0])
                    single_delta_w,single_delta_b=self.backprop(activations,Z,single_sample[0],single_sample[1])
                    delta_w=[dw+swd for dw,swd in zip(delta_w,single_delta_w)]
                    delta_b=[db+sdb for db,sdb in zip(delta_b,single_delta_b)]
                self.weights=[w-(self.eta)*dw/self.batch_size for w,dw in zip(self.weights,delta_w)]
                self.biases=[b-(self.eta)*db/self.batch_size for b,db in zip(self.biases,delta_b) ]

            #print(self.evaluate(test_data))
            res=0
            for single_sample in test_data:
                t_activation,t_Z=self.feed_forward(single_sample[0])
                #print single_sample[1]
                if single_sample[1]==np.argmax(t_activation[-1]):
                    res=res+1;
            print('Epoch ',e+1,'  :', res,'/10000')




    def backprop(self,activations,Z,sample_x,sample_y):
        delta_w=[np.zeros(np.shape(y)) for y in self.weights] 
        delta_b=[np.zeros(np.shape(y)) for y in self.biases]

        delta=(activations[-1]-sample_y)*self.diff_z(activations[-1])
        #print np.shape(delta)
        for l in xrange(2,len(self.sizes)):
            #print np.shape(delta)
            #print np.shape(delta_w[-l+1])
            #print np.shape(delta_b[-l+1])
            #print np.shape(self.weights[-l+1])
            delta_w[-l+1]=np.dot(delta,activations[-l].T)
            delta_b[-l+1]=delta 
            delta=np.dot(self.weights[-l+1].T,delta)*self.diff_z(Z[-l])

        return delta_w,delta_b
        


    def feed_forward(self,sample_x):
        activations=[sample_x]
        Z=[]
        #for y in xrange(len(self.sizes)):
        #   print np.shape(self.weights[y])
        for y in xrange(len(self.sizes)-1):
            #print np.shape(self.weights[y] )
            #print np.shape(sample_x)
            #print np.shape(self.biases[y])
            z=np.dot(self.weights[y],activations[y])+self.biases[y]
            Z.append(z)
            x=self.sigmoid(z);
            activations.append(x)
        return activations,Z

    def diff_z(self,a):
        return np.multiply(a,np.subtract(1,a))

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
