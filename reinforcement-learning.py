import pygame
import random
import numpy as np

pygame.init()

WIDTH, HEIGHT = 400, 300
AGENT_SIZE = 20
OBSTACLE_SIZE = 20
FPS = 30
NUM_EPISODES = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GOAL = (0, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning Park")

# Parámetros de Q-learning
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Discretización del espacio de estados
num_buckets = [10, 10, 2]  # [agent_x, agent_y, goal_reached]
state_bounds = [(0, WIDTH - AGENT_SIZE), (0, HEIGHT - AGENT_SIZE), (0, 1)]

# Inicializar la tabla Q con ceros
action_space_size = 4  # 0: move up, 1: move down, 2: move left, 3: move right
q_table = np.zeros(num_buckets + [action_space_size])

# Función de discretización del estado
def discretize_state(agent_x, agent_y, goal_reached):
    ratios = [agent_x / (state_bounds[0][1] - state_bounds[0][0]),
              agent_y / (state_bounds[1][1] - state_bounds[1][0]),
              goal_reached]
    discretized_state = [int(round((num_buckets[i] - 1) * ratios[i])) for i in range(3)]
    return tuple(discretized_state)

# Elegir una acción utilizando la estrategia epsilon-greedy
def choose_action(state, distance_to_goal):
    if np.random.uniform(0, 1) < epsilon or distance_to_goal < 20:
        return random.choice(range(action_space_size))  # explorar
    else:
        return np.argmax(q_table[state])  # explotar

# Calcular la distancia euclidiana entre dos puntos
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Obstaculos
def generate_obstacles():
    return [(random.randint(0, WIDTH - OBSTACLE_SIZE), random.randint(50, HEIGHT - OBSTACLE_SIZE - 50))
            for _ in range(15)]

def run_episode():
    agent_x, agent_y = WIDTH // 2, HEIGHT - AGENT_SIZE - 50  
    goal_x, goal_y = WIDTH - AGENT_SIZE - 40, 0  
    episode_steps = 0
    penalty = 0
    goal_reached = 0

    obstacles = generate_obstacles()
    obstacles.append((10, 10))  

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
                return episode_steps

        # Dibujar el parque
        screen.fill(WHITE)
        pygame.draw.rect(screen, GOAL, [goal_x, goal_y, AGENT_SIZE, AGENT_SIZE])

        # Mover y dibujar obstáculos
        for obstacle_pos in obstacles:
            obstacle_x, obstacle_y = obstacle_pos
            pygame.draw.rect(screen, BLACK, [obstacle_x, obstacle_y, OBSTACLE_SIZE, OBSTACLE_SIZE])

        # Carrito
        pygame.draw.rect(screen, RED, [agent_x, agent_y, AGENT_SIZE, AGENT_SIZE])

        pygame.display.flip()

        pygame.time.Clock().tick(FPS)

        # Incrementar pasos del episodio
        episode_steps += 1

        # Calcular la distancia al objetivo
        distance_to_goal = calculate_distance(agent_x, agent_y, goal_x, goal_y)

        # Elegir acción y mover el agente
        state = discretize_state(agent_x, agent_y, goal_reached)
        action = choose_action(state, distance_to_goal)

        # Mover el agente
        new_agent_x, new_agent_y = agent_x, agent_y
        if action == 0 and agent_y > 0:  # move up
            new_agent_y -= 5
        elif action == 1 and agent_y < HEIGHT - AGENT_SIZE:  # move down
            new_agent_y += 5
        elif action == 2 and agent_x > 0:  # move left
            new_agent_x -= 5
        elif action == 3 and agent_x < WIDTH - AGENT_SIZE:  # move right
            new_agent_x += 5

        # Restringir el movimiento vertical si el agente está en el límite superior
        if agent_y == 0:
            new_agent_y = agent_y  # No permitir movimiento hacia arriba

        # Comprobar colisiones con obstáculos
        collision = any(
            obstacle_x < new_agent_x + AGENT_SIZE and obstacle_x + OBSTACLE_SIZE > new_agent_x
            and obstacle_y < new_agent_y + AGENT_SIZE and obstacle_y + OBSTACLE_SIZE > new_agent_y
            for obstacle_x, obstacle_y in obstacles
        )

        # Actualizar la posición del agente si no hay colisión
        if not collision:
            agent_x, agent_y = new_agent_x, new_agent_y
        else:
            penalty += 1

        # Comprobar si el agente ha alcanzado la meta
        if agent_x == goal_x and agent_y == goal_y:
            goal_reached = 1
            print(f"Congratulations! Meta alcanzada en {episode_steps} pasos.")
            return episode_steps, penalty

# Ejecución de episodios
for episode in range(NUM_EPISODES):
    duration, penalty = run_episode()
    print(f"Episodio {episode + 1}: Duración = {duration} pasos, Penalizaciones = {penalty}")

pygame.quit()