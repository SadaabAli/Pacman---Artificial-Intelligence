# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

0,      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        closestFood = float('inf')
        for food in newFoodList:
            closestFood = min(closestFood,util.manhattanDistance(food,newPos))

        closestGhost = util.manhattanDistance(newGhostStates[0].getPosition(),newPos)
        score = successorGameState.getScore()
        if closestGhost > 0:
            score -= 10.0/closestGhost

        return score + 10.0/closestFood

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxScore = {}
        for action in gameState.getLegalActions(0):
            maxScore[action] = self.minFunction(gameState, gameState.generateSuccessor(0, action), self.depth,0)
        return max(maxScore.items(), key=lambda k: k[1])[0]

    def maxFunction(self,gameState,state,depth):
        if self.isTerminalState(depth,state):
            return self.evaluationFunction(state)

        value = float('-inf')
        for action in state.getLegalActions(0):
            value = max(value,self.minFunction(gameState,state.generateSuccessor(0,action),depth,0))
        return value

    def minFunction(self,gameState,state,depth,agent):
        if self.isTerminalState(depth,state):
            return self.evaluationFunction(state)

        agent = agent + 1
        value = float('inf')
        for action in state.getLegalActions(agent):
            if self.isGhost(agent,gameState):
                value = min(value, self.minFunction(gameState, state.generateSuccessor(agent, action), depth, agent))
            else:
                value = min(value, self.maxFunction(gameState, state.generateSuccessor(agent, action), depth-1))
        return value

    def isTerminalState(self,depth,state):
        return depth == 0 or state.isWin() or state.isLose()

    def isGhost(self,agent,gameState):
        return agent < (gameState.getNumAgents() - 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxScore = {}
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            newScore = self.minFunction(gameState, gameState.generateSuccessor(0, action), self.depth, 0, alpha, beta)
            maxScore[action] = newScore

            if newScore > beta:
                return max(maxScore.items(), key=lambda k: k[1])[0]
            alpha = max(alpha,newScore)
        return max(maxScore.items(), key=lambda k: k[1])[0]

    def maxFunction(self, gameState, state, depth, alpha, beta):
        if self.isTerminalState(depth, state):
            return self.evaluationFunction(state)

        value = float('-inf')
        for action in state.getLegalActions(0):
            newScore = self.minFunction(gameState, state.generateSuccessor(0, action), depth, 0, alpha, beta)
            value = max(value, newScore)
            if newScore > beta:
                return value
            alpha = max(alpha,newScore)
        return value

    def minFunction(self, gameState, state, depth, agent, alpha, beta):
        if self.isTerminalState(depth, state):
            return self.evaluationFunction(state)

        value = float('inf')
        agent = agent + 1
        for action in state.getLegalActions(agent):
            if self.isGhost(agent,gameState):
                newScore = self.minFunction(gameState, state.generateSuccessor(agent, action), depth, agent, alpha, beta)
                value = min(value, newScore)
            else:
                newScore = self.maxFunction(gameState, state.generateSuccessor(agent, action), depth - 1, alpha, beta)
                value = min(value, newScore)

            if newScore < alpha:
                return value
            beta = min(newScore,beta)
        return value

    def isTerminalState(self, depth, state):
        return depth == 0 or state.isWin() or state.isLose()

    def isGhost(self,agent,gameState):
        return agent < (gameState.getNumAgents() - 1)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        maxScore = {}
        for action in gameState.getLegalActions(0):
            maxScore[action] = self.expValueFunction(gameState, gameState.generateSuccessor(0, action), self.depth, 0)
        return max(maxScore.items(), key=lambda k: k[1])[0]

    def maxFunction(self, gameState, state, depth):
        if self.isTerminalState(depth, state):
            return self.evaluationFunction(state)

        value = float('-inf')
        for action in state.getLegalActions(0):
            value = max(value, self.expValueFunction(gameState, state.generateSuccessor(0, action), depth, 0))
        return value

    def expValueFunction(self, gameState, state, depth, agent):
        if self.isTerminalState(depth, state):
            return self.evaluationFunction(state)

        value = 0.0
        agent = agent + 1
        for action in state.getLegalActions(agent):
            if self.isGhost(agent,gameState):
                value += self.expValueFunction(gameState, state.generateSuccessor(agent, action), depth, agent)
            else:
                value += self.maxFunction(gameState, state.generateSuccessor(agent, action), depth - 1)
        return value / len(state.getLegalActions(agent))

    def isTerminalState(self, depth, state):
        return depth == 0 or state.isWin() or state.isLose()

    def isGhost(self,agent,gameState):
        return agent < (gameState.getNumAgents() - 1)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    evaluationValue = currentGameState.getScore()
    for ghost in newGhostStates:
        distanceToGhost = manhattanDistance(ghost.getPosition(),newPos)
        if distanceToGhost > 0:
            if ghost.scaredTimer > 0:
                evaluationValue += 1000.0 / distanceToGhost
            else:
                evaluationValue -= 100.0 / distanceToGhost

    foodDistance = [manhattanDistance(food,newPos) for food in newFood]
    if foodDistance:
        evaluationValue += 10.0 / min(foodDistance)

    return evaluationValue

# Abbreviation
better = betterEvaluationFunction

