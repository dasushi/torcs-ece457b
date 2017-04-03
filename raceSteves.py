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
import csv

from gym_torcs import TorcsEnv
from SteveTheActor import SteveTheActor
from SteveTheCritic import SteveTheCritic
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
ornUhlSteer = ornUhl(0.0, 0.15, 0.7)
#centered on 0.55 
ornUhlAcc = ornUhl(0.65, 0.1, 1.00) 
#centered on -0.1
ornUhlBrake = ornUhl(-0.1, 0.05, 0.90)

def playGame(trainFlag = 1):
    bufferLength = 50000
    #values from Google paper
    #http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    gamma = 0.99
    tau = 0.001
    epsilon = 1
    actorLearnRate = 0.0001
    criticLearnRate = 0.001
    epsilonDelta = 1 / 5000.0   
    episodeMax = 2500
    maxIter = 10000
    batchLength = 16
    step = 0
    flag = 0    
    totalLoss = 0
    complete = False
    actionDimensions = 3
    stateDimensions = 29
   
     
    #set TensorFlow to use GPU, add speedups 
    configProto = tensor.ConfigProto()
    configProto.gpu_options.allow_growth = True
    session = tensor.Session(config = configProto)
    K.set_session(session)
    
    #Initialize actor and critic
    actor = SteveTheActor(actionDimensions, stateDimensions, batchLength, 
                          session, actorLearnRate, tau)
    critic = SteveTheCritic(actionDimensions, stateDimensions, batchLength, 
                          session, criticLearnRate, tau)
    #initialize framebuffer for replays
    frameBuffer = FrameBuffer(bufferLength)
    #launch game
    torcsEnv = TorcsEnv(vision = False, throttle = True, gear_change = False)
    
    #load weights from file
    try:
        if os.path.isfile("steveActor.h5"):
            actor.model.load_weights("steveActor.h5")
            actor.targetModel.load_weights("steveActor.h5")
        if os.path.isfile("steveCritic.h5"):   
            critic.model.load_weights("steveCritic.h5")   
            critic.targetModel.load_weights("steveCritic.h5")
    except:
        print("Error loading weight files")
          
    for x in range(episodeMax):
        print("Start of Ep:" + str(x) + " Buffer:" + str(frameBuffer.getCount()))

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
            #perform action based on predicted input
            #get updated state information 
            observ, newReward, complete, info = torcsEnv.step(act[0])
            #stack new sensor information
            newStack = stackSensors(observ)
            #add new frame to frameBuffer
            frameBuffer.addFrame(stack, act[0], newReward, newStack, complete)
            #if frameBuffer.getSize() > batchLength: 
            batch = frameBuffer.getBatch(batchLength)
  
            state = np.asarray([i[0] for i in batch])
            actions = np.asarray([i[1] for i in batch])
            reward = np.asarray([i[2] for i in batch])
            newState = np.asarray([i[3] for i in batch])
            completeVector = np.asarray([i[4] for i in batch])
            yTrain = np.asarray([i[1] for i in batch])
            #yTrain = critic.getRewards(state, actions)
            #print(reward)
            #print(yTrain)
            targetQVal = critic.getRewards(newState, actor.targetModel.predict(newState))
            for z in range(len(batch)):
                #yTrain[z] = reward[z]
                if not completeVector[z]:
                    yTrain[z] = reward[z] + gamma * targetQVal[z]
                else:
                    yTrain[z] = reward[z]
                
            if(trainFlag):
                #update loss based on critic analyzing last action/state result
                loss += critic.model.train_on_batch([state, actions], yTrain)
                #actor predicts new input based on new state
                actorGradient = actor.model.predict(state)
                #critic is updated based on actor gradient result
                gradient = critic.gradients(state, actorGradient)
                
                #actor trained based on critic gradient
                actor.train(state, gradient)
                actor.trainTarget()
                critic.trainTarget()
                
            rewardSum = rewardSum + newReward
            stack = newStack
            #print("s:" + str(step) + " out:" + str(act) + " Loss:" + str(loss) + " Reward:" + str(rewardSum)) 
            step += 1
            totalLoss += loss
            if complete:
                break
        if trainFlag and np.mod(x, 5) == 0:
            print("Saving Actor and Critic Models")
            try:
                actor.model.save_weights("steveActor.h5", overwrite = True)
                with open("steveActor.json", "w") as actorFile:
                    dump(actor.model.to_json(), actorFile)

                critic.model.save_weights("steveCritic.h5", overwrite = True)
                with open("steveCritic.json", "w") as criticFile:
                    dump(critic.model.to_json(), criticFile)
            except:
                print("Error saving Actor and Critic Models")
        print("***Episode:" + str(x) + " Reward Sum:" + str(rewardSum) + "Loss:" + str(totalLoss))
        print("***Steps:" + str(step))
        with open('results.csv', 'a') as csvfile:
            wr = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL);
            wr.writerow([x, step, rewardSum, totalLoss, act])
            totalLoss = 0
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
