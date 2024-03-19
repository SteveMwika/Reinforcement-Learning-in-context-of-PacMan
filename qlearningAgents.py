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

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        # Initialize the Q-learning agent by calling the constructor of the parent class.
        ReinforcementAgent.__init__(self, **args)

        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        # a counter to store Q-values for state-action pairs.
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
         # Return Q-value for a given state-action pair.
        return self.qValues[(state, action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        # Get legal actions for the given state
        legalActions = self.getLegalActions(state)
        
        # If there are no legal actions, return a default value (0.0 in this case)s
        if not legalActions:
            return 0.0
          
        # Initialize an empty list to store Q values for legal actions
        q_values = []
        # Iterate over legal actions and calculate Q values
        for action in legalActions:
          q_value = self.getQValue(state, action)
          q_values.append(q_value)
          
        # Find the maximum Q value among the calculated values
        max_QValue = max(q_values)
        
        return max_QValue
      
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
          
        best_Actions = []
        
        for action in legalActions:
          # Check the Q-value for the current action is equal to the maximum Q-value for the state.
          if self.getQValue(state, action) == self.getValue(state):
              # If equal, add the action to the list of best actions.
              best_Actions.append(action)
              
        return random.choice(best_Actions)
      
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)
          

        # return action
        # util.raiseNotDefined()

        # return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        # Update the Q-value for the observed state-action pair using Q-learning update rule.
        # Calculate the sample value
        nextState_Value = self.computeValueFromQValues(nextState)
        sample = reward + self.discount * nextState_Value
        
        # Update the Q-value using the TD update formula
        previous_QValue = self.getQValue(state, action)
        updated_QValue = (1 - self.alpha) * previous_QValue + self.alpha * sample
        
        # Store the updated Q-value
        self.qValues[(state, action)] = updated_QValue
        
        #util.raiseNotDefined()

    def getPolicy(self, state):
      return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        # Extract features for the current state-action pair
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature in features:
            q_value += self.weights[feature] * features[feature]

        return q_value
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        # Extract features for the current state-action pair
        features = self.featExtractor.getFeatures(state, action)
        # Compute the Q-value for the next states
        nextQValue = self.computeValueFromQValues(nextState)
        # Calculate the temporal difference error
        difference = (reward + self.discount * nextQValue) - self.getQValue(state, action)
        
        # Update weights based on the features of the current state-action pair
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]
        
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** CS5368 Fall 2023 YOUR CODE HERE ***"
            print("Approximate Q-Agent Weights:")
            print(self.weights)
            #pass
