# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from collections import defaultdict
import random
import util
import math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
    """

    def __init__(self, **args):
        
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        #
        try:
            # if state in self.q_values and action in self.q_values[state]:
            return self.q_values[(state, action)]
            # else:
            # 0.0
        except Exception as e:
            print(e)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        #
        try:
            legal_actions = self.getLegalActions(state)
            if not legal_actions:
                return 0.0
            return max([self.getQValue(state, action) for action in legal_actions])
        except Exception as e:
            print(e)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  
        """
        try:
            legal_actions = self.getLegalActions(state)

            if not legal_actions:
                return None

            best_actions = {}
            max_q_value = float('-inf')

            for action in legal_actions:
                qvalue = self.getQValue(state, action)
                best_actions[action] = qvalue

                if qvalue > max_q_value:
                    max_q_value = qvalue

            bestActions = [act for act, qval in best_actions.items()
                           if qval == max_q_value]
            return random.choice(bestActions)
        except Exception as e:
            print(e)

    def getAction(self, state):
        """
          Compute the action to take in the current state. 
        """
        try:
            # Pick Action
            legalActions = self.getLegalActions(state)
            if len(legalActions) == 0:
                return 0
            action = None
            if not util.flipCoin(self.epsilon):
                action = self.getPolicy(state)
            else:
                action = random.choice(legalActions)
            return action
            #
        except Exception as e:
            print(e)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
         
        """
        #
        max_q_value = self.getValue(nextState)
        gamma = self.discount
        updated = reward + gamma * max_q_value
        self.q_values[(state, action)] += self.alpha * \
            (updated - self.q_values[(state, action)])

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """

        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
      
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          
        """
        #
        # return self.q_values[(state, action)]

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        # max_q_value = self.getValue(nextState)
        # gamma = self.discount
        # updated = reward + gamma * max_q_value
        # self.q_values[(state, action)] += self.alpha * \
        #     (updated - self.q_values[(state, action)])
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
