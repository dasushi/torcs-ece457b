import math
import numpy as np
import keras.backend as K
from keras import layers
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Input, Lambda, Flatten
from keras.initializers import RandomNormal, identity
from keras.optimizers import Adam

import tensorflow as tensor

#STEVE MCQUEEN is a bit of a split personality
#This is Steve McQueen the actor, who drives cars
#Uses a neural network with 2 hidden layers, 
HIDDEN_COUNT = 500

def normInit(shape, dtype=None):
        return K.random_normal(shape, dtype=dtype)

class SteveTheActor(object):
    def __init__(self, actionDimensions, stateDimensions, batchDimensions, session, learningRate, tau):
        #Initialize Steve's variables
        self.stateDimensions = stateDimensions
        self.actionDimensions = actionDimensions
        self.batchDimensions = batchDimensions
        self.learningRate = learningRate
        self.tau = tau
        
        self.session = session
        K.set_session(session)
        
        self.model, self.weights, self.state = self.initActorNetwork()
        self.targetModel, self.targetWeights, self.targetState = self.initActorNetwork()
      
        #create base action gradient
        self.action_gradient = tensor.placeholder(tensor.float32,[None, actionDimensions])
        #initialize parameter gradient based on inverse action gradient
        inverseAction = -self.action_gradient
        self.parameterGradient = tensor.gradients(self.model.output, self.weights, inverseAction)
 
        gradients = zip(self.parameterGradient, self.weights)
        self.optimize = tensor.train.AdamOptimizer(learningRate).apply_gradients(gradients)

        self.session.run(tensor.global_variables_initializer())

    def train(self, actionGrad, state):
        self.session.run(self.optimize, feed_dict={self.action_gradient: actionGrad,
                                                   self.state: state})

    def trainTarget(self):
        #update Actor's weights & target weights
        steveActorWeights = self.model.get_weights()
        steveActorTargetWeights = self.targetModel.get_weights()
       
        #target = actor * tau + (1 - tau) * target
        for i in range(len(steveActorWeights)):
            steveActorTargetWeights[i] = steveActorWeights[i] * self.tau + (1 - self.tau) * steveActorTargetWeights[i]
        #update new target weights
        self.targetModel.set_weights(steveActorTargetWeights)
        #Ring up Steve McQueen's agent and ask for his network
    def initActorNetwork(self):
        state = Input(shape = [self.stateDimensions])
        hidden0 = Dense(HIDDEN_COUNT, activation = 'relu')(state)
        hidden1 = Dense(2 * HIDDEN_COUNT, activation = 'relu')(hidden0)
        #Xavier weight initialization
        #Using Var(W) = 2 / neuronCount (output neurons approximate away) 
        variance = 2 / (3 * HIDDEN_COUNT)
        #randomStr = RandomNormal(mean = 0.0, stddev = variance, seed = 12345)
        #randomAcc = RandomNormal(mean = 0.5, stddev = variance, seed = 12345)
        #randomBrk = RandomNormal(mean = -0.1, stddev = variance, seed = 12345)
 
        #initialize control network signals
        #steering is [-1, 1] so it is tanh
        #others are [0, 1] so sigmoidal
        steeringControl = Dense(1, activation = 'tanh', init=normInit)(hidden1)#, kernel_initializer ='random_uniform', bias_initializer = 'zeros')(hidden1)
        accControl = Dense(1, activation = 'sigmoid', init=normInit)(hidden1)#, kernel_initializer = 'random_uniform', bias_initializer = 'zeros')(hidden1)
        brakeControl = Dense(1, activation = 'sigmoid', init=normInit)(hidden1)#, kernel_initializer = 'random_uniform', bias_initializer = 'zeros')(hidden1)
       
        concatLayer = layers.Concatenate(axis = 1)
        controlVector = concatLayer([steeringControl, accControl, brakeControl])
        totalModel = Model(state, controlVector)
        return totalModel, totalModel.trainable_weights, state




