import pygame
import math
import sys

# Define the colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255,255,255)

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Visualizer():
    def __init__(self,grid=(10,10), caption="Spec Ops Visualization", screen_dim=(400, 400)):
        # Initialize Pygame
        pygame.init()

        # Create the screen
        self.grid = grid
        self.screen_dim = (400, 400)
        self.screen = pygame.display.set_mode(self.screen_dim)

        # Set the caption
        self.caption = caption
        pygame.display.set_caption(self.caption)

        #Initialize Default State
        self.state = {
            "m1":{
                "species": "seal",
                "pos":{"x":0,"y":0},
                "angle":0,
                "status": "alive"
            },
            "t1":{
                "species": "terrorist",
                "pos":{"x":9,"y":9},
                "angle":0,
                "status": "alive"
            },
        }

    def update(self,state=None, reward={"soldier":0,"terrorist":0}):
        if(state == None): state = self.state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #Draw white background
        self.screen.fill(WHITE)

        #Draw the grid
        for i in range(0, self.screen_dim[0], int(self.screen_dim[0]/self.grid[0])):
            pygame.draw.line(self.screen, BLACK, (i, 0), (i, self.screen_dim[0]))
            pygame.draw.line(self.screen, BLACK, (0, i), (self.screen_dim[0], i))

        # Draw the agents
        for agent in state:
            agent = AttributeDict(state[agent])
            if(agent.species == 'seal'):
                a = 20
                pa = agent.angle
                pa = math.pi*pa/180
                px, py = (2*agent.pos['x']+1)*(self.screen_dim[0]/self.grid[0])/2, (2*agent.pos['y']+1)*(self.screen_dim[0]/self.grid[0])/2
                fx, fy = ((px+2*a*math.sin(pa),py+2*a*math.cos(pa)))
                signx = (-1,1)[(pa>=270 or pa<=90)]
                signy = (-1,1)[(pa>=180)]
                l = 1000
                fov = (agent.fov)*(math.pi/180)
                shoot_angle=(agent.shoot_angle)*(math.pi/180)
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa+fov/2), py+signy*l*math.sin(pa+fov/2)))
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa-fov/2), py+signy*l*math.sin(pa-fov/2)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(pa+shoot_angle/2), py+signy*l*math.sin(pa+shoot_angle/2)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(pa-shoot_angle/2), py+signy*l*math.sin(pa-shoot_angle/2)))
                pa += math.pi/2
                print((0,20*reward['soldier'],20+55*reward['soldier']))
                # pygame.draw.polygon(self.screen, (0,200*reward['soldier'],200+55*reward['soldier']), ((px+a*math.sin(pa),py+a*math.cos(pa)), (px+a*math.sin(math.pi/3-pa), py-a*math.cos(math.pi/3-pa)), (px-a*math.sin(2*math.pi/3-pa), py+a*math.cos(2*math.pi/3-pa))))
                pygame.draw.polygon(self.screen, (0,0,255), ((px+a*math.sin(pa),py+a*math.cos(pa)), (px+a*math.sin(math.pi/3-pa), py-a*math.cos(math.pi/3-pa)), (px-a*math.sin(2*math.pi/3-pa), py+a*math.cos(2*math.pi/3-pa))))
            elif(agent.species == 'terrorist'):
                a = 20
                pa = agent.angle
                pa = math.pi*pa/180
                px, py = (2*agent.pos['x']+1)*(self.screen_dim[0]/self.grid[0])/2, (2*agent.pos['y']+1)*(self.screen_dim[0]/self.grid[0])/2
                fx, fy = ((px+2*a*math.sin(pa),py+2*a*math.cos(pa)))
                signx = (-1,1)[(pa>270 or pa<90)]
                signy = (-1,1)[(pa>180)]
                l = 1000
                fov = (agent.fov)*(math.pi/180)
                shoot_angle=(agent.shoot_angle)*(math.pi/180)
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa+fov/2), py+signy*l*math.sin(pa+fov/2)))
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa-fov/2), py+signy*l*math.sin(pa-fov/2)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(pa+shoot_angle/2), py+signy*l*math.sin(pa+shoot_angle/2)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(pa-shoot_angle/2), py+signy*l*math.sin(pa-shoot_angle/2)))
                pa += math.pi/2
                print((20+55*reward['terrorist'],20*reward['soldier'],0))
                # pygame.draw.polygon(self.screen, (200+55*reward['terrorist'],200*reward['soldier'],0), ((px+a*math.sin(pa),py+a*math.cos(pa)), (px+a*math.sin(math.pi/3-pa), py-a*math.cos(math.pi/3-pa)), (px-a*math.sin(2*math.pi/3-pa), py+a*math.cos(2*math.pi/3-pa))))
                pygame.draw.polygon(self.screen, (255,0,0), ((px+a*math.sin(pa),py+a*math.cos(pa)), (px+a*math.sin(math.pi/3-pa), py-a*math.cos(math.pi/3-pa)), (px-a*math.sin(2*math.pi/3-pa), py+a*math.cos(2*math.pi/3-pa))))

        # Update the screen
        pygame.display.flip()

    def quit(self):
        pygame.quit()
