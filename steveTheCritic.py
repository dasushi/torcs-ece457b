import math
import numpy as np
import keras.backend as K
from keras import layers
from keras.models import Sequential, load_model, model_from_json, Model
from keras.layers import Dense, Input, Lambda, Flatten, Activation
from keras.layers.merge import Add
from keras.initializers import RandomNormal, identity
from keras.optimizers import Adam

import tensorflow as tensor

#STEVE MCQUEEN is a bit of a split personality
#This is Steve McQueen the actor, who drives cars
#Uses a neural network with 2 hidden layers, 
HIDDEN_COUNT = 500


class SteveTheCritic(object):
    def __init__(self, actionDimensions, stateDimensions, batchDimensions, session, learningRate,tau):
        #Initialize Steve's variables
        self.stateDimensions = stateDimensions
        self.actionDimensions = actionDimensions
        self.batchDimensions = batchDimensions
        self.learningRate = learningRate
        self.tau = tau
        
        self.session = session
        K.set_session(session)
        
        self.model, self.action, self.state = self.initCriticNetwork()
        self.targetModel, self.targetAction, self.targetState = self.initCriticNetwork()
      
        self.action_gradients = tensor.gradients(self.model.output, self.action)
        self.session.run(tensor.global_variables_initializer())

    def gradients(self, action, state):
        result = self.session.run(self.action_gradients, 
                                  feed_dict = {
                                                  self.state: state, 
                                                  self.action: action
                                              })
        return result[0]


    def trainTarget(self):
        #update Critic's weights & target weights
        steveCriticWeights = self.model.get_weights()
        steveCriticTargetWeights = self.targetModel.get_weights()
       
        #target = actor * tau + (1 - tau) * target
        for i in range(len(steveCriticWeights)):
            steveCriticTargetWeights[i] = steveCriticWeights[i] * self.tau + (1 - self.tau) * steveCriticTargetWeights[i]
        #update with new target weights
        self.targetModel.set_weights(steveCriticTargetWeights)
    
    def initCriticNetwork(self):
        action = Input(shape = [self.actionDimensions])
        state = Input(shape = [self.stateDimensions])
        postAction = Dense(2 * HIDDEN_COUNT, activation = 'linear')(action)
        weight = Dense(HIDDEN_COUNT, activation = 'relu')(state)
        hidden = Dense(2 * HIDDEN_COUNT, activation = 'relu')(weight)
        #Sum the hidden layer input and action input
        sumHidden = layers.add([hidden, postAction])
        #take sum down to a linear action output
        denseHidden = Dense(2 * HIDDEN_COUNT, activation = 'relu')(sumHidden)
        criticVector = Dense(self.actionDimensions, activation = 'linear')(denseHidden) 
        
        totalModel = Model([action, state], criticVector)
        totalModel.compile(loss = 'mse', optimizer = Adam(lr = self.learningRate))
        return totalModel, action, state




