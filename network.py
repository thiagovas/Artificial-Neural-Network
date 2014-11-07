# By Thiago Silva

from random import uniform
from math import exp


class network:
    """A class that represents a artificial neural network"""

    weightsXToHidden = []
    weightsYToHidden = []
    weightsHiddenToOutput = []
    biasesHidden = []
    hiddenValues = []
    biasOutput = 1
    out = 0
    ro = 0.2
    euler = exp(1)

    def __init__(self):
        self.weightsXToHidden = [uniform(0, 1) for x in range(6)]
        self.weightsYToHidden = [uniform(0, 1) for x in range(6)]
        self.weightsHiddenToOutput = [uniform(0, 1) for x in range(6)]
        self.hiddenValues = [uniform(0, 1) for x in range(6)]
        self.biasesHidden = [1 for x in range(6)]
        self.biasOutput = 1
        self.euler = exp(1)
        self.ro = 0.2


    def squash(self, value):
        '''
        This function is used to keep values in the interval[0, 1]
        :param value: Value used as a parameter in the squashing function.
        :return: Return the value after applied to the function 1/(1+e^(-value))
        '''
        return 1.0/(1.0+self.euler**(-value))


    def getOutput(self, inputX, inputY):
        '''
        :param inputX: The value of the first input
        :param inputY: The value of the second input
        :return: Return the value that the neural network returns given inputX and inputY as inputs.
        '''
        self.out = self.biasOutput

        for i in range(6):
            self.hiddenValues[i] = self.weightsXToHidden[i]*inputX
        for i in range(6):
            self.hiddenValues[i] += self.weightsYToHidden[i]*inputY+self.biasesHidden[i]
        for i in range(6):
            self.hiddenValues[i] = self.squash(self.hiddenValues[i])
        for i in range(6):
            self.out += self.hiddenValues[i]*self.weightsHiddenToOutput[i]
        self.out = self.squash(self.out)
        return self.out


    def updateWeights(self, inputX, inputY, desiredOut):
        '''
        This function update the weights of the network.
        :param inputX: The X value of the exemplar input
        :param inputY: The Y value of the exemplar input
        :param desiredOut: Given inputX, and inputY the desired output.
        :return: This function returns the delta of the output node
        '''
        deltaOut = self.out*(1-self.out)*(desiredOut-self.out)
        deltasHidden = []

        for i in range(len(self.weightsHiddenToOutput)):
            self.weightsHiddenToOutput[i] += self.ro*deltaOut*self.hiddenValues[i]
        self.biasOutput += self.ro*deltaOut

        for i in range(len(self.hiddenValues)):
            deltasHidden.append(self.hiddenValues[i]*(1-self.hiddenValues[i])*deltaOut*self.weightsHiddenToOutput[i])

        for i in range(len(self.weightsXToHidden)):
            self.weightsXToHidden[i] += self.ro*deltasHidden[i]*inputX
            self.weightsYToHidden[i] += self.ro*deltasHidden[i]*inputY
            self.biasesHidden[i] += self.ro*deltasHidden[i]

        return deltaOut

