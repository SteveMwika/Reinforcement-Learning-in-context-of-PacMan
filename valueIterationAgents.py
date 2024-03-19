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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        # Initializing instance variables
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        states = mdp.getStates()

        # Write value iteration code here
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        # Value iteration loop
        for _ in range(iterations):
            newValues = util.Counter()  # Temporary storage for updated values
            for state in mdp.getStates():
                if not mdp.isTerminal(state):
                    # If the state is not terminal, update its value
                    newValues[state] = max(
                        self.computeQValueFromValues(state, action)
                        for action in mdp.getPossibleActions(state)
                    )
            self.values = newValues  # Update values after each iteration
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state] # Returning the value of the specified state


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        qValue = 0.0 # Initializing the Q-value to 0

        # Iterate over possible next states and their probabilities
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # Getting the immediate reward
            reward = self.mdp.getReward(state, action, nextState) 
            # Discounted future value
            discountedFutureValue = self.discount * self.values[nextState]  
            # Updating Q-value based on probabilities
            qValue += prob * (reward + discountedFutureValue) 

        return qValue # Returning computed Q-value
    
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** CS5368 Fall 2023 YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None  # No legal actions at terminal state

         # Getting possible actions
        possible_Actions = self.mdp.getPossibleActions(state)
        best_Action = self.getBestAction(possible_Actions, state)
        return best_Action # Return best action according to the policy

    def getBestAction(self, possible_Actions, state):
        """
        Helper method to get the action with the maximum Q-value
        from a list of possible actions for the given state.
        """
        #lambda function (a concise way to create small anonymous functions). It takes an action as an argument and returns the corresponding Q-value for that action in the current state.
        # Finding the action with the maximum Q-value using a lambda function
        bestAction = max(possible_Actions, key=lambda action: self.computeQValueFromValues(state, action))
        return bestAction
        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state) # Returns best action according to the policy

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        # Returns the Q-value for the specified state-action pair
        return self.computeQValueFromValues(state, action)
