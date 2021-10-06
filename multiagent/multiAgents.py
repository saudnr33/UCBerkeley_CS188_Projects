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

def absDiff(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(scores, legalMoves, chosenIndex)

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

        ## Food Calculationns:
        ClosestGhost = absDiff(newPos, successorGameState.getGhostPositions()[0])
        ClosestGhost_pos = (0, 0)
        for g in successorGameState.getGhostPositions():
            Temp = absDiff(newPos, g)
            if Temp <= ClosestGhost:
                ClosestGhost = Temp
                ClosestGhost_pos = g

        if ClosestGhost_pos == newPos:
            return -500


        foodList = currentGameState.getFood().asList()
        if len(foodList) != 0:

            ClosestFood = absDiff(newPos, foodList[0])

            if currentGameState.getFood()[newPos[0]][newPos[1]]:
                ClosestFood = 0
            else:
                for f in foodList:
                    Temp = absDiff(newPos, f)
                    if Temp <= ClosestFood:
                        ClosestFood = Temp

            return -  1 / (ClosestGhost) + 1 / (ClosestFood +1)
        return 500


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
        hh = self.minmax(gameState, self.depth, 0)[0]
        #print(hh)
        return hh
        util.raiseNotDefined()




    def minmax(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return "s" , self.evaluationFunction(gameState)
        if agentIndex == 0:
            actions = gameState.getLegalActions(agentIndex)
            maxValue = -1e1000
            optimalAction = ""

            # random.shuffle(actions)
            #print(gameState.getNumAgents())

            for a in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, a)

                eval = self.minmax(nextGameState, depth , 1)[1]

                if eval > maxValue:
                    maxValue = eval
                    optimalAction = a
            return optimalAction, maxValue
        else:
            actions = gameState.getLegalActions(agentIndex)
            minValue = 1e1000
            optimalAction = ""

            # random.shuffle(actions)
            for a in actions:


                nextGameState = gameState.generateSuccessor(agentIndex, a)
                if agentIndex == gameState.getNumAgents() - 1:

                    eval = self.minmax(nextGameState, depth - 1, (agentIndex+1)% gameState.getNumAgents())[1]
                else:
                    eval = self.minmax(nextGameState, depth, (agentIndex+1)% gameState.getNumAgents())[1]

                if eval < minValue:
                    minValue = eval
                    optimalAction = a
            return optimalAction, minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        hh = self.minmax(gameState, self.depth, 0, alpha = -1e10000, beta = 1e10000)[0]
        #print(hh)
        return hh
        util.raiseNotDefined()




    def minmax(self, gameState, depth, agentIndex, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return "s" , self.evaluationFunction(gameState)


        if agentIndex == 0:
            actions = gameState.getLegalActions(agentIndex)
            maxValue = -1e1000
            optimalAction = ""


            #print(gameState.getNumAgents())

            for a in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, a)

                eval = self.minmax(nextGameState, depth , 1, alpha, beta)[1]
                alpha = max(alpha, eval)
                if eval > maxValue:
                    maxValue = eval
                    optimalAction = a

                if beta < alpha:
                    break
            return optimalAction, maxValue
        else:
            actions = gameState.getLegalActions(agentIndex)
            minValue = 1e1000
            optimalAction = ""


            for a in actions:


                nextGameState = gameState.generateSuccessor(agentIndex, a)
                if agentIndex == gameState.getNumAgents() - 1:

                    eval = self.minmax(nextGameState, depth - 1, (agentIndex+1)% gameState.getNumAgents(), alpha, beta)[1]
                else:
                    eval = self.minmax(nextGameState, depth, (agentIndex+1)% gameState.getNumAgents(), alpha, beta)[1]

                if eval < minValue:
                    minValue = eval
                    optimalAction = a

                beta = min(beta, eval)
                if beta < alpha:
                    break
            return optimalAction, minValue


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
        hh = self.minmax(gameState, self.depth, 0)[0]
        #print(hh)
        return hh
        util.raiseNotDefined()




    def minmax(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return "s" , self.evaluationFunction(gameState)
        if agentIndex == 0:
            actions = gameState.getLegalActions(agentIndex)
            maxValue = -1e1000
            optimalAction = ""

            # random.shuffle(actions)
            #print(gameState.getNumAgents())

            for a in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, a)

                eval = self.minmax(nextGameState, depth , 1)[1]

                if eval > maxValue:
                    maxValue = eval
                    optimalAction = a
            return optimalAction, maxValue
        else:
            actions = gameState.getLegalActions(agentIndex)
            minValue = 1e1000
            optimalAction = ""

            # random.shuffle(actions)
            toReturn = 0
            for a in actions:


                nextGameState = gameState.generateSuccessor(agentIndex, a)
                if agentIndex == gameState.getNumAgents() - 1:

                    eval = self.minmax(nextGameState, depth - 1, (agentIndex+1)% gameState.getNumAgents())[1]
                else:
                    eval = self.minmax(nextGameState, depth, (agentIndex+1)% gameState.getNumAgents())[1]

                toReturn += eval
            return "", toReturn/len(actions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    minFood = 10000
    for f in currentGameState.getFood().asList():
        minFood = min(minFood, manhattanDistance(pos, f))

    minCap = 10000
    for c in currentGameState.getCapsules():
        minCap = min(minCap, manhattanDistance(pos, c))

    minG = 10000
    for g in currentGameState.getGhostStates():
        Temp = manhattanDistance(pos, g.getPosition())
        if Temp < minG:
            if g.scaredTimer >= Temp:
                minG = 1/(Temp + 1)
            else:
                minG = Temp

    a, b, c, d, e, f= 1/5, 11, 3, 16, 8, 12
    return  (a * (minG + minFood)) + (b * currentGameState.getScore()) + (c * sum(scaredTimes)) + (d/ (len( currentGameState.getCapsules()) + minFood)) + ((e / (minCap + 1)) + (f / (minFood + 1)))


# Abbreviation
better = betterEvaluationFunction
