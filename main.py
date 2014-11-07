#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Thiago Silva

from network import network
from itertools import permutations


def main():
    '''
    Main method:
        Here I read the input, and for
    '''
    net = network()
    n = int(input())
    inputX = []
    inputY = []
    outputs = []
    totalError = 0

    for i in range(n):
        inp = [int(x) for x in raw_input().split(' ')]
        inputX.append(inp[0])
        inputY.append(inp[1])
        outputs.append(inp[2])

    cont = 0
    # Each loop is an epoch.
    # While there isn't a set of weights on the neural network with error less than 0.4 the loop doesnt break.
    for currentOrder in permutations(range(n)):
        outs = []
        for i in currentOrder:
            outs.append(net.getOutput(inputX[i], inputY[i])) # Calculating the output
            net.updateWeights(inputX[i], inputY[i], outputs[i]) # Updating the weights of the Neural network
		
        totalError = 0
        goingToBreak = True
        for i in currentOrder:
            totalError += (outputs[i]-outs[i])**2
            if abs(outputs[i]-outs[i]) > 0.4:
                goingToBreak = False
        if goingToBreak:
            break
    
        if cont % 10 == 0:
            print('Epoch ' + str(cont))
            print('Squared Error: ' + str(totalError) + '\n')
        cont += 1

    delta = 0
    for i in range(n):
        o = net.getOutput(inputX[i], inputY[i])
        delta += abs(outputs[i]-o)
        print('Exemplar: ' + str(inputX[i]) + ' ' + str(inputY[i]) + ' ' + str(outputs[i]) + '   Neural Network Output: ' + str(o))
    print('\ndelta: ' + str(delta/8))



if __name__ == "__main__":
	main()
