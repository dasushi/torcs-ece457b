import math
import numpy as np
import keras.backend as K
from keras import layers
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Input, Lambda, Flatten
from keras.initializers import RandomNormal, identity
from keras.optimizers import Adam
from keras.objectives import mean_squared_error

import tensorflow as tensor

#STEVE MCQUEEN is a bit of a split personality
#This is Steve McQueen the actor, who drives cars
#Uses a neural network with 2 hidden layers, 
HIDDEN_COUNT = 400


#Xavier weight initialization
#Using Var(W) = 2 / neuronCount (output neurons approximate away) 
#variance = 2 / (3 * HIDDEN_COUNT)
def normInit(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype, stddev=(2 / 3 * HIDDEN_COUNT))

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
        def initActorNetwork(stateDim):
            state = Input(shape = [stateDim])
            hidden0 = Dense(HIDDEN_COUNT, activation = 'relu', init='lecun_uniform')(state)
            hidden1 = Dense(2 * HIDDEN_COUNT, activation = 'relu', init='lecun_uniform')(hidden0)

            #initialize control network signals
            #steering is [-1, 1] so it is tanh
            #others are [0, 1] so sigmoidal
            steeringControl = Dense(1, activation = 'tanh',init=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(hidden1)
            accControl = Dense(1, activation = 'sigmoid',init=RandomNormal(mean=0.0, stddev=1e-4,seed=None))(hidden1)
            brakeControl = Dense(1, activation = 'sigmoid',init=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(hidden1)       
            controlVector = layers.concatenate([steeringControl, accControl, brakeControl])
            #controlVector = concatLayer([steeringControl, accControl, brakeControl])
            totalModel = Model(inputs=state, outputs=controlVector)
            #totalModel.compile(loss='mse', optimizer=Adam(lr =self.learningRate)) 
            return totalModel, totalModel.trainable_weights, state, controlVector


        self.model, self.weights, self.state, self.action = initActorNetwork(stateDimensions)
        self.targetModel, self.targetWeights, self.targetState, self.targetAction = initActorNetwork(stateDimensions)
      
        #create base action gradient
        action_gradient = tensor.placeholder(tensor.float32,[None, actionDimensions])
        #combine parameter gradient based on inverse action gradient
        self.weight_grad = tensor.gradients(self.action, self.weights, -action_gradient)
        #self.loss = tensor.reduce_mean(mean_squared_error(-self.action_gradient, self.model.output)) 
        gradients = zip(self.weight_grad, self.weights)
        self.optimize = tensor.train.AdamOptimizer(learningRate).apply_gradients(gradients)
        init = tensor.global_variables_initializer()
        self.action_gradient = action_gradient
        self.session.run(init)

    
    def train(self, state, actionGrad):
        self.session.run(self.optimize, feed_dict={ self.state: state, 
                                                    self.action_gradient: actionGrad })

    def get_actions(self, states):
        return self.targetModel.predict(states)


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



