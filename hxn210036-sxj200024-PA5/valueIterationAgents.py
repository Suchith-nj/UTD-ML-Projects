# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp 
import util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        try:
            for i in range(self.iterations):
                new_values = {}

                for state in self.mdp.getStates():
                    if self.mdp.isTerminal(state):
                        new_values[state] = 0
                    else:
                        new_values[state] = max(self.getQValue(state, action) for action in self.mdp.getPossibleActions(state))
                self.values = new_values
            
            policy = {}
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    policy[state] = None
                else:
                    
                    best_action = None
                    best_value = float('-inf')
                    for action in self.mdp.getPossibleActions(state):
                        
                        q_value = self.getQValue(state, action)
                        if q_value > best_value:
                            best_value = q_value
                            best_action = action
                    policy[state] = best_action

            self.policy = policy
            
        except Exception as e:
            print(e)

    def getValue(self, state):
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        try:
            Qvalues = list()
            transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)  # [(s_prime, prob), ...]
            
            for nextState, prob in transitionStates:
                currReward = self.mdp.getReward(state, action, nextState)
                discountFactor = self.discount
                nextVal = self.values[nextState]
                # df= self.discount if self.discount else 0.0
                # df = max(discountFactor, df)
                gamma = discountFactor * nextVal
                q = prob * (currReward+gamma)
                Qvalues.append(q)
            return sum(Qvalues)
        except Exception as e:
            print(e)


    def computeActionFromValues(self, state):

        try:

            actions = self.mdp.getPossibleActions(state)

            if len(actions) != 0:

                QvalsActions = list()  # [(action, qval) , ......]

                for action in actions:
                    qval = self.getQValue(state, action)
                    QvalsActions.append((action, qval))

                maxQvalAction = max(QvalsActions, key=lambda x: x[1])
                best_action = maxQvalAction[0]
                return best_action
            else:
                return None
        except Exception as e:
            print(e)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

