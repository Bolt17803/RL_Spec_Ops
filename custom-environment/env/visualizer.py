import pygame
import math
import sys

# Define the colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0,128,0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
DIM_GREY = (105, 105, 105)
DIM = (81, 72, 79)
soldier_sprite = pygame.image.load('Maps/soldier.png')
terrorist_sprite = pygame.image.load('Maps/terrorist.png')
block = pygame.image.load('Maps/block.png')
path = pygame.image.load('Maps/path.png')
dil = pygame.image.load('Maps/dil.png')
width = 55
height = 55
soldier = pygame.transform.scale(soldier_sprite, (width, height))
terrorist = pygame.transform.scale(terrorist_sprite, (width, height))
block = pygame.transform.scale(block, (55, 55))
path = pygame.transform.scale(path, (55, 55))
dil = pygame.transform.scale(dil,(5,5))
bullets = []


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Visualizer():
    def __init__(self, grid=(10, 10), caption="Spec Ops Visualization", screen_dim=(825,825), agents=None):
        # Initialize Pygame
        pygame.init()

        # Create the screen
        self.grid = grid
        self.screen_dim = screen_dim
        self.screen = pygame.display.set_mode(self.screen_dim)
        self.screen.fill(DIM_GREY)
        # Set the caption
        self.caption = caption
        pygame.display.set_caption(self.caption)

        # Initialize agents
        self.agents = agents
        if (self.agents is None):
            print("VISUALIZER ERROR: No agents given in Visualizer!!")
            exit()

    def update(self, state=None, reward={"soldier": 0, "terrorist": 0}):
        if (state is None):
            print("VISUALIZER ERROR: No state given to render!!")
            exit()
        health_bar_width = 40  # Define health bar width
        health_bar_height = 5  # Define health bar height
        max_health = 2  # Maximum health for soldier and terrorist
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw white background
        self.screen.fill(DIM_GREY)

        # Draw the grid
        w = int(self.screen_dim[0] / self.grid[0])
        for i in range(0, self.screen_dim[0], int(self.screen_dim[0] / self.grid[0])):
            for j in range(0, self.screen_dim[1], int(self.screen_dim[1] / self.grid[1])):
                self.screen.blit(path, (i, j))

        # Draw walls
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                if state['map'][j][i] == 4:
                    self.screen.blit(block, (w * i, w * j, w, w))

        # Draw the agents
        for agent in self.agents:
            agent_name = agent
            agent = AttributeDict(state[agent])

            if 'soldier' in agent_name:
                # Set the length of an arrow to represent soldier orientation
                a = 20
                # Convert the angle from degrees to radians
                pa = agent.angle
                pa = math.pi * pa / 180
                # Calculate the position of the soldier
                px, py = (2 * agent.x + 1) * (self.screen_dim[0] / self.grid[0]) / 2, (
                            2 * agent.y + 1) * (self.screen_dim[0] / self.grid[0]) / 2
                # Calculate the endpoint of the arrow (direction the soldier is facing)
                fx, fy = ((px + 2 * a * math.sin(pa), py + 2 * a * math.cos(pa)))
                # Determine the direction of the arrow based on the soldier's angle
                signx = (-1, 1)[(pa >= 270 or pa <= 90)]
                signy = (-1, 1)[(pa >= 180)]
                # Set the length of the arrow
                l = 1000
                # Calculate soldier's field of view (fov) and shoot angle
                fov = (agent.fov) * (math.pi / 180)
                shoot_angle = (agent.shoot_angle) * (math.pi / 180)
                # Draw lines representing the soldier's field of view
                pygame.draw.line(self.screen, BLACK, (px, py),
                                 (px + signx * l * math.cos(pa + fov / 2), py + signy * l * math.sin(pa + fov / 2)))
                pygame.draw.line(self.screen, BLACK, (px, py),
                                 (px + signx * l * math.cos(pa - fov / 2), py + signy * l * math.sin(pa - fov / 2)))
                # Draw lines representing the soldier's shooting angle
                pygame.draw.line(self.screen, RED, (px, py),
                                 (px + signx * l * math.cos(pa + shoot_angle / 2),
                                  py + signy * l * math.sin(pa + shoot_angle / 2)))
                pygame.draw.line(self.screen, RED, (px, py),
                                 (px + signx * l * math.cos(pa - shoot_angle / 2),
                                  py + signy * l * math.sin(pa - shoot_angle / 2)))
                # Increment the angle for further drawing
                pa += math.pi / 2
                if 'hp' in agent:
                    sol_hp = agent['hp']
                    health_percentage = sol_hp / 2  # Calculate health percentage
                    # ... (existing code for health bar drawing)

                    # Draw damage indicator if health is reduced
                    if sol_hp < 2:  # Check if health reduced
                        self.screen.blit(dil, (agent.x * self.screen_dim[0] / self.grid[0],
                                                                 agent.y * self.screen_dim[1] / self.grid[1]))

                    # Draw health bar for the soldier
                    health_bar_x = agent.x * self.screen_dim[0] / self.grid[0]  # X position of health bar
                    health_bar_y = agent.y * self.screen_dim[1] / self.grid[1] - 10  # Y position of health bar
                    remaining_health_width = int(health_percentage * health_bar_width)

                    pygame.draw.rect(self.screen, GREEN, (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
                    pygame.draw.rect(self.screen, RED, (health_bar_x + remaining_health_width, health_bar_y,
                                                        health_bar_width - remaining_health_width, health_bar_height))

                    # Rotate the soldier image based on its angle
                    rotated_soldier = pygame.transform.rotate(soldier, agent.angle)
                    # Calculate the position to blit the rotated image (centered at the agent's position)
                    rotated_rect = rotated_soldier.get_rect(center=(agent.x * self.screen_dim[0] / self.grid[0] + w / 2,
                                                                    agent.y * self.screen_dim[1] / self.grid[1] + w / 2))
                    self.screen.blit(rotated_soldier, rotated_rect.topleft)
                
                # # Rotate the soldier image based on its angle
                rotated_soldier = pygame.transform.rotate(soldier, agent.angle)
                # # Calculate the position to blit the rotated image (centered at the agent's position)
                
                rotated_rect = rotated_soldier.get_rect(center=(w * agent.x + w / 2, w * agent.y + w / 2))

                self.screen.blit(rotated_soldier, rotated_rect.topleft)
               
            elif 'terrorist' in agent_name:
                # Check if the agent is a terrorist
                a = 20  # Set the length of an arrow to represent terrorist orientation
                pa = agent.angle  # Get the angle of the terrorist
                pa = math.pi * pa / 180  # Convert the angle from degrees to radians
                # Calculate the position of the terrorist on the screen
                px, py = (2 * agent.x + 1) * (self.screen_dim[0] / self.grid[0]) / 2, (
                            2 * agent.y + 1) * (self.screen_dim[0] / self.grid[0]) / 2
                # Calculate the endpoint of the arrow (direction the terrorist is facing)
                fx, fy = ((px + 2 * a * math.sin(pa), py + 2 * a * math.cos(pa)))
                # Determine the direction of the arrow based on the terrorist's angle
                signx = (-1, 1)[(pa > 270 or pa < 90)]
                signy = (-1, 1)[(pa > 180)]
                l = 1000  # Set the length for some visual elements
                fov = (agent.fov) * (math.pi / 180)  # Calculate terrorist's field of view (fov)
                shoot_angle = (agent.shoot_angle) * (math.pi / 180)  # Calculate the shooting angle
                # Draw lines representing the terrorist's field of view
                pygame.draw.line(self.screen, BLACK, (px, py),
                                 (px + signx * l * math.cos(pa + fov / 2), py + signy * l * math.sin(pa + fov / 2)))
                pygame.draw.line(self.screen, BLACK, (px, py),
                                 (px + signx * l * math.cos(pa - fov / 2), py + signy * l * math.sin(pa - fov / 2)))
                # Draw lines representing the terrorist's shooting angle
                pygame.draw.line(self.screen, RED, (px, py),
                                 (px + signx * l * math.cos(pa + shoot_angle / 2),
                                  py + signy * l * math.sin(pa + shoot_angle / 2)))
                pygame.draw.line(self.screen, RED, (px, py),
                                 (px + signx * l * math.cos(pa - shoot_angle / 2),
                                  py + signy * l * math.sin(pa - shoot_angle / 2)))
                pa += math.pi / 2  # Increment the angle for further drawing    
                if 'hp' in agent:
                    ter_hp = agent['hp']
                    health_percentage = ter_hp / 2  # Calculate health percentage
                    # ... (existing code for health bar drawing)

                    # Draw damage indicator if health is reduced
                    if ter_hp < 2:  # Check if health reduced
                        self.screen.blit(dil, (agent.x * self.screen_dim[0] / self.grid[0],
                                                                 agent.y * self.screen_dim[1] / self.grid[1]))

                    # Draw health bar for the soldier
                    health_bar_x = agent.x * self.screen_dim[0] / self.grid[0]  # X position of health bar
                    health_bar_y = agent.y * self.screen_dim[1] / self.grid[1] - 10  # Y position of health bar
                    remaining_health_width = int(health_percentage * health_bar_width)

                    pygame.draw.rect(self.screen, GREEN, (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
                    pygame.draw.rect(self.screen, RED, (health_bar_x + remaining_health_width, health_bar_y,
                                                        health_bar_width - remaining_health_width, health_bar_height))

                    # Rotate the soldier image based on its angle
                    rotated_terrorist = pygame.transform.rotate(terrorist, agent.angle)
                    # Calculate the position to blit the rotated image (centered at the agent's position)
                    rotated_rect = rotated_terrorist.get_rect(center=(agent.x * self.screen_dim[0] / self.grid[0] + w / 2,
                                                                    agent.y * self.screen_dim[1] / self.grid[1] + w / 2))
                    self.screen.blit(rotated_terrorist, rotated_rect.topleft)
                # Rotate the terrorist image based on its angle
                rotated_terrorist = pygame.transform.rotate(terrorist, agent.angle)
                # Calculate the position to blit the rotated image (centered at the agent's position)
                rotated_rect = rotated_terrorist.get_rect(center=(agent.x * self.screen_dim[0] / self.grid[0] + w / 2,
                                                                  agent.y * self.screen_dim[1] / self.grid[1] + w / 2))
                self.screen.blit(rotated_terrorist, rotated_rect.topleft)

                # Draw the terrorist image at the calculated position
                rotated_terrorist = pygame.transform.rotate(terrorist, agent.angle)
                # Calculate the position to blit the rotated image (centered at the agent's position)
                rotated_rect = rotated_terrorist.get_rect(center=(w * agent.x + w / 2, w * agent.y + w / 2))
                self.screen.blit(rotated_terrorist, rotated_rect.topleft)

        pygame.display.flip()

    def quit(self):
        pygame.quit()

