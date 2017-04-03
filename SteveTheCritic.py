import math
import numpy as np
import keras.backend as K
from keras import layers
from keras.models import Sequential, load_model, model_from_json, Model
from keras.layers import Dense, Input, Lambda, Flatten, Activation
from keras.layers.merge import Add
from keras.initializers import RandomNormal, identity
from keras.optimizers import Adam
from keras.objectives import mean_squared_error

import tensorflow as tensor

#STEVE MCQUEEN is a bit of a split personality
#This is Steve McQueen the actor, who drives cars
#Uses a neural network with 2 hidden layers, 
HIDDEN_COUNT = 400


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
        def initCriticNetwork(stateDim, actionDim):
            #state input
            state = Input(name = "state", shape = [stateDim])
            weight = Dense(HIDDEN_COUNT, activation = 'relu')(state)
            #action input
            action = Input(name = "action", shape = [actionDim])
            postAction = Dense(2 * HIDDEN_COUNT, activation = 'linear')(action)

            #state hidden layer
            hidden = Dense(2 * HIDDEN_COUNT, activation = 'linear')(weight)
            #Sum the hidden layer input and action input
            sumHidden = layers.add([hidden, postAction])
            #take sum down to a linear action output
            denseHidden = Dense(2 * HIDDEN_COUNT, activation = 'relu')(sumHidden)
            criticVector = Dense(actionDim, activation = 'linear')(denseHidden) 
        
            totalModel = Model(inputs=[state, action], outputs=criticVector)
            opt = Adam(lr = self.learningRate)
            totalModel.compile(loss = 'mse', optimizer = opt)
            return totalModel, action, state



        self.model, self.action, self.state = initCriticNetwork(stateDimensions, actionDimensions)
        self.targetModel, self.targetAction, self.targetState = initCriticNetwork(stateDimensions, actionDimensions)
        #self.predictedQ = tensor.placeholder(tensor.float32, [None,actionDimensions])
        #self.loss = tensor.reduce_mean(mean_squared_error(self.predictedQ, self.model.output))
        #self.optimize = tensor.train.AdamOptimizer(self.learningRate).minimize(self.loss) 
        self.action_gradients = tensor.gradients(self.model.output, self.action)
      
        self.session.run(tensor.global_variables_initializer())

    def gradients(self, states, actions):
        result = self.session.run(self.action_gradients, 
                                  feed_dict = { self.state: states, 
                                                self.action: actions })
        #print(result[0])
        return result[0]

    def train(self, states, actions, targets):
        return self.model.fit([states, actions], targets, nb_epoch=1, verbose=False)
   
    def getRewards(self, states, actions):
        return self.targetModel.predict([states, actions])

    def trainTarget(self):
        #update Critic's weights & target weights
        steveCriticWeights = self.model.get_weights()
        steveCriticTargetWeights = self.targetModel.get_weights()
       
        #target = actor * tau + (1 - tau) * target
        for i in range(len(steveCriticWeights)):
            steveCriticTargetWeights[i] = steveCriticWeights[i] * self.tau + (1 - self.tau) * steveCriticTargetWeights[i]
        #update with new target weights
        self.targetModel.set_weights(steveCriticTargetWeights)
    



