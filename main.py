import pygame
import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# define some colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# set the size of the window
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400

# set the size and speed of the snake
BLOCK_SIZE = 20

# initialize the game
pygame.init()
CLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
FONT = pygame.font.Font(None, 50)

class Snake:
    # define actions
    ACTIONS = [0, 1, 2, 3]

    def __init__(self):
        self.length = 3
        self.positions = [((WINDOW_WIDTH // 2), (WINDOW_HEIGHT // 2))]
        self.direction = random.choice(self.ACTIONS)
        self.color = GREEN
        self.score = 0

    def get_head_position(self):
        return self.positions[0]

    def turn(self, new_direction):
        # Don't allow the snake to move in the opposite direction instantaneously
        opposing_directions = [(0, 1), (1, 0), (2, 3), (3, 2)]
        for direction in opposing_directions:
            if (new_direction == direction[0] and self.direction == direction[1]) or (
                    new_direction == direction[1] and self.direction == direction[0]):
                return

        self.direction = new_direction

    def move(self):
        cur = self.get_head_position()
        if self.direction == 0:  # previously pygame.K_UP
            new_pos = (cur[0], cur[1] - BLOCK_SIZE)
        elif self.direction == 1:  # previously pygame.K_DOWN
            new_pos = (cur[0], cur[1] + BLOCK_SIZE)
        elif self.direction == 2:  # previously pygame.K_LEFT
            new_pos = (cur[0] - BLOCK_SIZE, cur[1])
        else:  # previously pygame.K_RIGHT
            new_pos = (cur[0] + BLOCK_SIZE, cur[1])

        if len(self.positions) > self.length:
            self.positions.pop()

        self.positions.insert(0, new_pos)

    def draw(self):
        for p in self.positions:
            pygame.draw.rect(SCREEN, self.color, pygame.Rect(p[0], p[1], BLOCK_SIZE, BLOCK_SIZE))

    def get_state(self, food):
        head_x, head_y = self.get_head_position()
        food_x, food_y = food.position

        state = [
            # Position of the snake's head and the food
            head_x,
            head_y,
            food_x,
            food_y,

            # Current direction of the snake
            self.ACTIONS.index(self.direction)
        ]

        return state

    def is_collision(snake):
        head_x, head_y = snake.get_head_position()
        return (head_x < 0 or head_y < 0 or head_x >= WINDOW_WIDTH or head_y >= WINDOW_HEIGHT or
                (snake.get_head_position() in snake.positions[1:]))

    def get_reward(self, food_eaten, game_over):
        if game_over:
            return -100  # Big negative reward if the game is over
        elif food_eaten:
            return 100  # Big positive reward if the snake eats the food
        else:
            return -1  # negative reward for every action that doesn't lead to food

    def update_score(self, reward):
        self.score += reward

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, (WINDOW_WIDTH - BLOCK_SIZE) // BLOCK_SIZE - 1) * BLOCK_SIZE,
                         random.randint(0, (WINDOW_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE - 1) * BLOCK_SIZE)

    def draw(self):
        pygame.draw.rect(SCREEN, self.color, pygame.Rect(self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_name):
        self.model.save_weights(file_name)

    def load_model(self, file_name):
        self.model.load_weights(file_name)



def main():
    # create agent only once
    agent = DQNAgent(5, 4)  # 4 possible directions
    scores = []  # list to store scores
    max_score = -np.inf  # Initialize maximum score to negative infinity

    try:
        agent.load_model("deep_learn_database/dqn_model.h5")
        print("Lesssssssgo!!!")
    except FileNotFoundError:
        print("No model found. Starting from scratch.")

    # try:
    #     agent.load_model("deep_learn_database/best_dqn_model.h5")
    #     print("Lesssssssgo with only best!!!")
    # except FileNotFoundError:
    #     print("No model found.")

    # start the learning loop
    for episode in range(10000):   # replace NUM_EPISODES with desired number of learning episodes

        # reset the game state
        snake = Snake()
        food = Food()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            old_state = snake.get_state(food)
            action = agent.act(old_state)
            snake.turn(action)
            snake.move()

            SCREEN.fill(BLACK)

            snake.draw()
            food.draw()

            food_eaten = snake.get_head_position() == food.position
            game_over = Snake.is_collision(snake)

            reward = snake.get_reward(food_eaten, game_over)
            snake.update_score(reward)

            if food_eaten:
                snake.length += 1
                food.randomize_position()

            if snake.score > max_score:
                max_score = snake.score
                agent.save_model("deep_learn_database/best_dqn_model.h5")

            if game_over:
                if episode % 10 == 0:  # print every 10 episodes
                    print(f'Episode {episode}, Score: {snake.score}, Max score: {max_score}')
                scores.append(snake.score)  # append the score
                snake.score = 0
                break

            new_state = snake.get_state(food)
            agent.remember(old_state, action, reward, new_state, game_over)

            pygame.display.update()
            CLOCK.tick(1000)

        agent.epsilon_decay *= agent.epsilon_decay

    try:
        agent.save_model("deep_learn_database/dqn_model.h5")
        print("Saved!")
    except FileNotFoundError:
        print("Not saved :(")



    # After the training loop
    plt.plot(scores)
    plt.title('Training Scores Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()


if __name__ == "__main__":
    main()