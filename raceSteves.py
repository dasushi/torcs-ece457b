import numpy as np
import tensorflow as tensor
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras import backend as K
from json import dump
import random
import argparse
import timeit
import os

from gym_torcs import TorcsEnv
from steveTheActor import SteveTheActor
from steveTheCritic import SteveTheCritic
from FrameBuffer import FrameBuffer

np.random.seed(14245)    

class ornUhl(object):
    def __init__(self, mu, sigma, theta):
        self.sigma = sigma
        self.mu = mu
        self.theta = theta
    def calc(self, x):
        return np.random.randn(1) * self.sigma + self.theta * (self.mu - x)
#centered on 0
ornUhlSteer = ornUhl(0.0, 0.15, 0.6)
#centered on 0.55 
ornUhlAcc = ornUhl(0.5, 0.1, 1.00) 
#centered on -0.1
ornUhlBrake = ornUhl(-0.1, 0.05, 1.00)

def playGame(trainFlag = 1):
    bufferLength = 100000
    #values from Google paper
    #http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    gamma = 0.99
    tau = 0.001
    epsilon = 1
    actorLearnRate = 0.0001
    criticLearnRate = 0.001
    epsilonDelta = 1 / 100000.0   
    episodeMax = 1000
    maxIter = 100000
    batchLength = 32
    step = 0
    flag = 0    
    complete = False
    actionDimensions = 3
    stateDimensions = 29
   

    
    configProto = tensor.ConfigProto()
    configProto.gpu_options.allow_growth = True
    session = tensor.Session(config = configProto)
    K.set_session(session)

    actor = SteveTheActor(actionDimensions, stateDimensions, batchLength, 
                          session, actorLearnRate, tau)
    critic = SteveTheCritic(actionDimensions, stateDimensions, batchLength, 
                          session, criticLearnRate, tau)
    
    frameBuffer = FrameBuffer(bufferLength)
    torcsEnv = TorcsEnv(vision = False, throttle = True, gear_change = False)
    
    try:
        if os.path.isfile("steveActor.h5"):
            actor.model.load_weights("steveActor.h5")
        if os.path.isfile("steveActor.h5"):   
            actor.target_model.load_weights("steveActor.h5")
        if os.path.isfile("steveCritic.h5"):   
            critic.model.load_weights("steveCritic.h5")
        if os.path.isfile("steveCritic.h5"):   
            critic.target_model.load_weights("steveCritic.h5")
    except:
        print("Error loading weight files")
          
    for x in range(episodeMax):
        print("Start of Ep:" + str(x) + " Buffer: " + str(frameBuffer.getCount()))

        if np.mod(x, 5) == 0:
            observ = torcsEnv.reset(relaunch = True)
        else:
            observ = torcsEnv.reset()

        stack = stackSensors(observ)
        rewardSum = 0.0
        for y in range(maxIter):
            epsilon = epsilon - epsilonDelta

            act = np.zeros([1, actionDimensions])
            loss = 0
            actPredict = actor.model.predict(stack.reshape(1, stack.shape[0]))
            
            #During training, apply Orn-Uhl noise to generate variance
            if trainFlag:
                noise = calcNoise(actionDimensions, actPredict, epsilon)
                act[0][0] = actPredict[0][0] + noise[0][0]
                act[0][1] = actPredict[0][1] + noise[0][1]
                act[0][2] = actPredict[0][2] + noise[0][2]
            else:
                act[0][0] = actPredict[0][0]
                act[0][1] = actPredict[0][1]
                act[0][2] = actPredict[0][2]
            #perform action based on predicted input, get updated state information 
            observ, newReward, complete, info = torcsEnv.step(act[0])
            #stack new sensor information
            newStack = stackSensors(observ)
            #add new frame to frameBuffer
            frameBuffer.addFrame(stack, act[0], newReward, newStack, complete)
            
            batch = frameBuffer.getBatch(batchLength)
  
            state = np.asarray([i[0] for i in batch])
            actions = np.asarray([i[1] for i in batch])
            yTrain = actions
            reward = np.asarray([i[2] for i in batch])
            newState = np.asarray([i[3] for i in batch])
            completeVector = np.asarray([i[4] for i in batch])
            #print(state)
            predictControls = actor.targetModel.predict(newState)
            #print(predictControls)
            targetQVal = critic.targetModel.predict([predictControls, newState])
            
            for z in range(len(batch)):
                yTrain[z] = reward[z]
                if not completeVector[z]:
                    yTrain[z] = yTrain[z] + gamma * targetQVal[z]
                
            if(trainFlag):
                #update loss based on critic analyzing last action/state result
                loss = loss + critic.model.train_on_batch([actions, state], yTrain)
                #actor predicts new input based on new state
                actorGradient = actor.model.predict(state)
                #print(actorGradient)
                #print(state)
                #critic is updated based on actor gradient result
                gradients = critic.gradients(actorGradient, state)
                #actor trained based on critic gradient
                actor.train(gradients, state)
                actor.trainTarget()
                critic.trainTarget()
                
            rewardSum = rewardSum + newReward
            stack = newStack
            print("Done Ep: " + str(x) + " Step: " + str(step) + " act: " + str(act) + " loss: " + str(loss) + " Reward: " + str(newReward)) 
            step += 1
            if complete:
                break
        if np.mod(x, 5) == 0:
            if(trainFlag):
                print("Saving Actor and Critic Models")
                actor.model.save_weights("steveActor.h5", overwrite = True)
                with open("steveActor.json", "w") as actorFile:
                    dump(actor.model.to_json(), actorFile)

                critic.model.save_weights("steveCritic.h5", overwrite = True)
                with open("steveCritic.json", "w") as criticFile:
                    dump(critic.model.to_json(), criticFile)
        print("Sum Episode: " + str(x) + " Reward Sum: " + str(rewardSum))
        print("Steps: " + str(step))
    torcsEnv.end()
    print("Race Ended!")

def calcNoise(actionDimensions, actOrig, epsilon):
    noise = np.zeros([1, actionDimensions])
    noise[0][0] = ornUhlSteer.calc(actOrig[0][0]) * max(epsilon, 0)
    noise[0][1] = ornUhlAcc.calc(actOrig[0][1]) * max(epsilon, 0)
    noise[0][2] = ornUhlBrake.calc(actOrig[0][2]) * max(epsilon, 0)
    return noise 
                   
def stackSensors(observ):
    return np.hstack((observ.angle,
                      observ.track, 
                      observ.trackPos,
                      observ.speedX,
                      observ.speedY,
                      observ.speedZ,
                      observ.wheelSpinVel / 100.0,
                      observ.rpm))

if __name__ == "__main__":
    playGame()            
