import random
from collections import deque

class FrameBuffer(object):
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.buffer = deque()
        self.count = 0

    def getSize(self):
        return self.bufferSize
    
    def getCount(self):
        return self.count

    #add one frame of state observations & other frame information
    def addFrame(self, state, action, reward, newState, complete):
        frame = (state, action, reward, newState, complete)
        if(self.count < self.bufferSize):
            self.buffer.append(frame)
            self.count = self.count + 1

        else:
            self.buffer.popleft()
            self.buffer.append(frame)
            
    def getBatch(self, batchSize):
        if self.count < batchSize:
            return random.sample(self.buffer, self.count)
        else: 
            return random.sample(self.buffer, batchSize)

    def empty(self):
        self.buffer = deque()
        self.count = 0
