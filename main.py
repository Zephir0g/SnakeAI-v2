import pygame
import random
import numpy as np
import os

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
    ACTIONS = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]

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
        opposing_directions = [(pygame.K_UP, pygame.K_DOWN), (pygame.K_DOWN, pygame.K_UP), (pygame.K_LEFT, pygame.K_RIGHT), (pygame.K_RIGHT, pygame.K_LEFT)]
        for direction in opposing_directions:
            if (new_direction == direction[0] and self.direction == direction[1]) or (new_direction == direction[1] and self.direction == direction[0]):
                return

        self.direction = new_direction

    def move(self):
        cur = self.get_head_position()
        if self.direction == pygame.K_UP:
            new_pos = (cur[0], cur[1] - BLOCK_SIZE)
        elif self.direction == pygame.K_DOWN:
            new_pos = (cur[0], cur[1] + BLOCK_SIZE)
        elif self.direction == pygame.K_LEFT:
            new_pos = (cur[0] - BLOCK_SIZE, cur[1])
        else:
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
            return -1
        elif food_eaten:
            return 1
        else:
            return 0

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

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor

        # Initialize Q-table to zeros. For simplicity, we consider
        # the state as the difference in x and y (each ranging from -10 to 10)
        # between the snake and the food, and the current direction of the snake.
        self.q_table = np.zeros((21, 21, 21, 21, 4, 4))

        # Define the action index mapping
        self.action_index_mapping = {
            pygame.K_UP: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_RIGHT: 3,
        }

    def get_max_q_value_action(self, state):
        head_x, head_y, food_x, food_y, direction = self.state_to_index(state)

        # Get the action that has the maximum Q-value
        max_q_value_action_index = np.argmax(self.q_table[head_x, head_y, food_x, food_y, direction])
        return self.index_to_action(max_q_value_action_index)

    def update_q_table(self, state, action, reward, next_state):
        head_x, head_y, food_x, food_y, direction = self.state_to_index(state)
        action_index = self.action_index_mapping[action]

        # Get the maximum Q-value for the next state
        next_head_x, next_head_y, next_food_x, next_food_y, next_direction = self.state_to_index(next_state)
        max_next_q_value = np.max(self.q_table[next_head_x, next_head_y, next_food_x, next_food_y, next_direction])

        # Update the Q-value
        self.q_table[head_x, head_y, food_x, food_y, direction, action_index] = (1 - self.alpha) * self.q_table[head_x, head_y, food_x, food_y, direction, action_index] + self.alpha * (reward + self.gamma * max_next_q_value)

    def state_to_index(self, state):
        head_x, head_y, food_x, food_y, direction = state
        return min(head_x // BLOCK_SIZE, 20), min(head_y // BLOCK_SIZE, 20), min(food_x // BLOCK_SIZE, 20), min(
            food_y // BLOCK_SIZE, 20), direction

    def index_to_action(self, action_index):
        # Convert the action index to the actual action
        return list(self.action_index_mapping.keys())[action_index]

    def get_action(self, state):
        # Choose action according to the epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return random.choice(Snake.ACTIONS)
        else:
            return self.get_max_q_value_action(state)



def main():
    # create agent only once
    agent = QLearningAgent()

    # start the learning loop
    for episode in range(500):   # replace NUM_EPISODES with desired number of learning episodes

        # reset the game state
        snake = Snake()
        food = Food()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            old_state = snake.get_state(food)
            action = agent.get_action(old_state)
            snake.turn(action)
            snake.move()

            SCREEN.fill(WHITE)

            snake.draw()
            food.draw()

            food_eaten = snake.get_head_position() == food.position
            game_over = Snake.is_collision(snake)

            reward = snake.get_reward(food_eaten, game_over)
            snake.update_score(reward)

            if food_eaten:
                snake.length += 1
                food.randomize_position()

            if game_over:
                if episode % 10 == 0: # print every 10 episodes
                    print(f'Episode {episode}, Score: {snake.score}')
                # Reset score after logging
                snake.score = 0

                text = FONT.render("Game Over", True, BLACK)
                SCREEN.blit(text,
                            (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - text.get_height() // 2))
                pygame.display.update()
                pygame.time.wait(1)  # delay to see game over message
                break

            new_state = snake.get_state(food)
            agent.update_q_table(old_state, action, reward, new_state)

            pygame.display.update()
            CLOCK.tick(240)


if __name__ == "__main__":
    main()
