import pygame
import math
import sys

# Define the colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255,255,255)
PINK = (235,32,239)
LIGHT_GREEN = (144, 238, 144)  # Define light green color
soldier_sprite = pygame.image.load('Maps/soldier.png')
terrorist_sprite = pygame.image.load('Maps/terrorist.png')
width = 40
height = 40
soldier = pygame.transform.scale(soldier_sprite, (width, height))
terrorist = pygame.transform.scale(terrorist_sprite, (width, height))
class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Visualizer():
    def __init__(self,grid=(10,10), caption="Spec Ops Visualization", screen_dim=(800, 800), agents=None):
        # Initialize Pygame
        pygame.init()

        # Create the screen
        self.grid = grid
        self.screen_dim = screen_dim
        self.screen = pygame.display.set_mode(self.screen_dim)
        self.screen.fill(LIGHT_GREEN)
        # Set the caption
        self.caption = caption
        pygame.display.set_caption(self.caption)

        #Initialize agents
        self.agents = agents
        if(self.agents == None):
            print("VISUALIZER ERROR: No agents given in Visualizer!!")
            exit()

    def update(self,state=None, reward={"soldier":0,"terrorist":0}):
        if(state == None):
            print("VISUALIZER ERROR: No state given to render!!")
            exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #Draw white background
        self.screen.fill(WHITE)

        #Draw the grid
        w = int(self.screen_dim[0]/self.grid[0])
        for i in range(0, self.screen_dim[0], int(self.screen_dim[0]/self.grid[0])):
            pygame.draw.line(self.screen, LIGHT_GREEN, (i, 0), (i, self.screen_dim[0]))
            pygame.draw.line(self.screen, LIGHT_GREEN, (0, i), (self.screen_dim[0], i))

        #Draw walls
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                # print(i,j)
                if state['map'][j][i] == 4:
                    pygame.draw.rect(self.screen,PINK,(w*i,w*j,w,w))

        # Draw the agents
        for agent in self.agents:
            #print(state[agent])
            agent_name = agent
            agent = AttributeDict(state[agent])
            #print(agent)
            if 'soldier' in agent_name:
    # Set the length of an arrow to represent soldier orientation
                a = 20
                # Convert the angle from degrees to radians
                pa = agent.angle
                pa = math.pi * pa / 180
                # Calculate the position of the soldier
                px, py = (2 * agent.x + 1) * (self.screen_dim[0] / self.grid[0]) / 2, (2 * agent.y + 1) * (self.screen_dim[0] / self.grid[0]) / 2
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
                pygame.draw.line(self.screen, BLACK, (px, py), (px + signx * l * math.cos(pa + fov / 2), py + signy * l * math.sin(pa + fov / 2)))
                pygame.draw.line(self.screen, BLACK, (px, py), (px + signx * l * math.cos(pa - fov / 2), py + signy * l * math.sin(pa - fov / 2)))
                # Draw lines representing the soldier's shooting angle
                pygame.draw.line(self.screen, RED, (px, py), (px + signx * l * math.cos(pa + shoot_angle / 2), py + signy * l * math.sin(pa + shoot_angle / 2)))
                pygame.draw.line(self.screen, RED, (px, py), (px + signx * l * math.cos(pa - shoot_angle / 2), py + signy * l * math.sin(pa - shoot_angle / 2)))
                # Increment the angle for further drawing
                pa += math.pi / 2
                # Draw a polygon representing the soldier's view
                # Draw a polygon representing the soldier's body
                # pygame.draw.polygon(self.screen, (0, 0, 255),
                #                     ((px + a * math.sin(pa), py + a * math.cos(pa)),
                #                     (px + a * math.sin(math.pi / 3 - pa), py - a * math.cos(math.pi / 3 - pa)),
                #                     (px - a * math.sin(2 * math.pi / 3 - pa), py + a * math.cos(2 * math.pi / 3 - pa))))
                # Draw a rectangle representing the soldier's position on the grid
                # pygame.draw.rect(self.screen,BLACK,(w*agent.x,w*agent.y,w,w))
                # Draw the soldier image at the calculated position
                self.screen.blit(soldier, (w * agent.x, w * agent.y))
            elif 'terrorist' in agent_name:
    # Check if the agent is a terrorist
                a = 20  # Set the length of an arrow to represent terrorist orientation
                pa = agent.angle  # Get the angle of the terrorist
                pa = math.pi * pa / 180  # Convert the angle from degrees to radians
                # Calculate the position of the terrorist on the screen
                px, py = (2 * agent.x + 1) * (self.screen_dim[0] / self.grid[0]) / 2, (2 * agent.y + 1) * (self.screen_dim[0] / self.grid[0]) / 2
                # Calculate the endpoint of the arrow (direction the terrorist is facing)
                fx, fy = ((px + 2 * a * math.sin(pa), py + 2 * a * math.cos(pa)))
                # Determine the direction of the arrow based on the terrorist's angle
                signx = (-1, 1)[(pa > 270 or pa < 90)]
                signy = (-1, 1)[(pa > 180)]
                l = 1000  # Set the length for some visual elements
                fov = (agent.fov) * (math.pi / 180)  # Calculate terrorist's field of view (fov)
                shoot_angle = (agent.shoot_angle) * (math.pi / 180)  # Calculate the shooting angle
                # Draw lines representing the terrorist's field of view
                pygame.draw.line(self.screen, BLACK, (px, py), (px + signx * l * math.cos(pa + fov / 2), py + signy * l * math.sin(pa + fov / 2)))
                pygame.draw.line(self.screen, BLACK, (px, py), (px + signx * l * math.cos(pa - fov / 2), py + signy * l * math.sin(pa - fov / 2)))
                # Draw lines representing the terrorist's shooting angle
                pygame.draw.line(self.screen, RED, (px, py), (px + signx * l * math.cos(pa + shoot_angle / 2), py + signy * l * math.sin(pa + shoot_angle / 2)))
                pygame.draw.line(self.screen, RED, (px, py), (px + signx * l * math.cos(pa - shoot_angle / 2), py + signy * l * math.sin(pa - shoot_angle / 2)))
                pa += math.pi / 2  # Increment the angle for further drawing
                # Draw a polygon representing the terrorist's view
                # pygame.draw.polygon(self.screen, (255, 0, 0),
                #                     ((px + a * math.sin(pa), py + a * math.cos(pa)),
                #                     (px + a * math.sin(math.pi / 3 - pa), py - a * math.cos(math.pi / 3 - pa)),
                #                     (px - a * math.sin(2 * math.pi / 3 - pa), py + a * math.cos(2 * math.pi / 3 - pa))))
		        # Draw the terrorist image at the calculated position
                self.screen.blit(terrorist, (w * agent.x, w * agent.y))
        # Update the screen
        pygame.display.flip()

    def quit(self):
        pygame.quit()
        
