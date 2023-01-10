import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple  # Assigning a meaning to each position in the tuple allowing a more readable code

pygame.init()  # Initialising all the modules
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):  # Creating a class Enum in order to create the correct direction
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# Constants
BSIZE = 20
SPEED = 100

# RGB Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (154, 50, 205)
GREEN = (127, 255, 0)
PINK = (252, 223, 225)
DBLUE = (0, 0, 139)
BBLUE = (0, 255, 255)


class SnakeGame:

    def __init__(self, w=640, h=480):  # The size of the window in which the game will be played
        self.w = w
        self.h = h
        # Initialising the display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Initialising the game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)  # Choosing the point in the middle of the screen
        self.snake = [self.head,
                      Point(self.head.x - BSIZE, self.head.y),  # Creating the snake
                      Point(self.head.x - (2 * BSIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._food_position()
        self.frame_iteration = 0

    def _food_position(self):  # Getting random positions on the screen for the food
        x = random.randint(0, (self.w - BSIZE) // BSIZE) * BSIZE
        y = random.randint(0, (self.h - BSIZE) // BSIZE) * BSIZE  
        self.food = Point(x, y)
        if self.food in self.snake:
            self._food_position()

    def play_step(self, action):
        self.frame_iteration += 1
        # Collecting the user input data
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Movement
        self._move(action)  # Updating the head
        self.snake.insert(0, self.head)  # Inserting the head in the snake

        # Checking if the game is over
        reward = 0
        game_over = False
        if self.collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Placing the food or just moving the snake
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._food_position()
        else:
            self.snake.pop()

        # Updating the pygame UI and the clock
        self._update_ui()
        self.clock.tick(SPEED)
        # Return if game over and return score
        return reward, game_over, self.score

    # Collision function
    def collision(self, pt = None):
        if pt is None:
            pt = self.head
        # Hits the boundary
        if pt.x > self.w - BSIZE or pt.x < 0 or pt.y > self.h - BSIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(PINK)
        for pt in self.snake:
            # Drawing the snake
            pygame.draw.rect(self.display, DBLUE, pygame.Rect(pt.x, pt.y, BSIZE, BSIZE))
            pygame.draw.rect(self.display, BBLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Drawing the food
        pygame.draw.rect(self.display, PURPLE, pygame.Rect(self.food.x, self.food.y, BSIZE, BSIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x + 4, self.food.y + 4, 12, 12))

        Stext = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(Stext, [0, 0])
        pygame.display.flip()

    # Movement function
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[index]   # no change of direction
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = clock_wise[next_index] # right turn
        else:
            next_index = (index - 1) %4
            new_direction = clock_wise[next_index] # left turn

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BSIZE
        elif self.direction == Direction.LEFT:
            x -= BSIZE
        elif self.direction == Direction.UP:
            y -= BSIZE
        elif self.direction == Direction.DOWN:
            y += BSIZE

        self.head = Point(x, y)
