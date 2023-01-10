import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeGame, Direction, Point
from Model import Linear_QNet, QTrainer
from Helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 32
LEARNING_RATE = 0.001


class Agent:

    def __init__(self):
        self.game_nr = 0
        self.epsilon = 0     # A parameter used to control the randomness
        self.gamma = 0.9      # Discount rate           
        self.memory = deque(maxlen = MAX_MEMORY)     # If we exceed MAX_MEMORY, popleft() will be called
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LEARNING_RATE, gamma = self.gamma)

    # Calculating the state while we are aware of the environment
    def get_state(self, game):  
        head = game.snake[0]
        point_r = Point(head.x + 20, head.y)
        point_l = Point(head.x - 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # An array with 11 states
        state = [
            # Danger straight
            (dir_r and game.collision(point_r)) or 
            (dir_l and game.collision(point_l)) or 
            (dir_u and game.collision(point_u)) or 
            (dir_d and game.collision(point_d)),

            # Danger to the right
            (dir_u and game.collision(point_r)) or 
            (dir_d and game.collision(point_l)) or 
            (dir_l and game.collision(point_u)) or 
            (dir_r and game.collision(point_d)),

            # Danger to the left
            (dir_d and game.collision(point_r)) or 
            (dir_u and game.collision(point_l)) or 
            (dir_r and game.collision(point_u)) or 
            (dir_l and game.collision(point_d)),
            
            # Move direction
            dir_r,
            dir_l,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x > game.head.x,  # food to the right
            game.food.x < game.head.x,  # food to the left
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype = int)       # Converting a boolean into an integer
        
    # We calculate the next move from the state
    def remember(self, state, action, reward, new_state, game_over):
        self. memory.append((state, action, reward, new_state, game_over))       # popleft()


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)       # It will return a list of tuple
        else:
            mini_sample = self.memory
        # Putting the data together
        states, actions, rewards, new_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, new_states, game_overs)

    def train_short_memory(self, state, action, reward, new_state, game_over):
        self.trainer.train_step(state, action, reward, new_state, game_over)
    
    def get_action(self, state):
        # The tradeoff between exploration and exploitation
        self.epsilon = 80 - self.game_nr
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # Getting the current state
        old_state = agent.get_state(game)

        # Getting the move
        final_move = agent.get_action(old_state)

        # Performing the move
        reward, game_over, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # Training the short memory only for one step
        agent.train_short_memory(old_state, final_move, reward, new_state, game_over)

        # Storing the data.
        agent.remember(old_state, final_move, reward, new_state, game_over)

        if game_over:
            # Training the long memory. Improves the AI in time.
            game.reset()
            agent.game_nr += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.game_nr, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_nr
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
