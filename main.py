import pygame
import random

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
    def __init__(self):
        self.length = 3
        self.positions = [((WINDOW_WIDTH // 2), (WINDOW_HEIGHT // 2))]
        self.direction = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT])
        self.color = GREEN

    def get_head_position(self):
        return self.positions[0]

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

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, WINDOW_WIDTH // BLOCK_SIZE - 1) * BLOCK_SIZE, random.randint(0, WINDOW_HEIGHT // BLOCK_SIZE - 1) * BLOCK_SIZE)

    def draw(self):
        pygame.draw.rect(SCREEN, self.color, pygame.Rect(self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

def is_collision(snake):
    head_x, head_y = snake.get_head_position()
    return head_x < 0 or head_y < 0 or head_x >= WINDOW_WIDTH or head_y >= WINDOW_HEIGHT

def main():
    snake = Snake()
    food = Food()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                snake.direction = event.key

        snake.move()

        SCREEN.fill(WHITE)

        snake.draw()
        food.draw()

        if snake.get_head_position() == food.position:
            snake.length += 1
            food.randomize_position()

        if is_collision(snake):
            text = FONT.render("Game Over", True, BLACK)
            SCREEN.blit(text, (WINDOW_WIDTH//2 - text.get_width()//2, WINDOW_HEIGHT//2 - text.get_height()//2))
            pygame.display.update()
            pygame.time.wait(3000)  # delay to see game over message
            return

        pygame.display.update()
        CLOCK.tick(30)

if __name__ == "__main__":
    main()
