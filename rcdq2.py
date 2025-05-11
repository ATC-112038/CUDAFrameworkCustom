import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

def init():
    pygame.init()
    display = (500, 400)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.set_caption("cuda-outp")  # Changed window title here
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glEnable(GL_DEPTH_TEST)

def cube():
    vertices = [
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    glBegin(GL_LINES)
    for i, edge in enumerate(edges):
        glColor3fv((0, 1, 0) if i < 8 else (1, 1, 1))
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for vertex in vertices:
        glColor3fv((1, 0, 0))
        glVertex3fv((0, 0, 0))
        glVertex3fv(vertex)
    glEnd()
    
    glPointSize(5)
    glBegin(GL_POINTS)
    for vertex in vertices:
        glColor3fv((1, 0, 0))
        glVertex3fv(vertex)
    glEnd()

def main():
    init()
    clock = pygame.time.Clock()
    rotation_x, rotation_y, rotation_z = 0, 0, 0
    font = pygame.font.SysFont('Arial', 20)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        rotation_x += 1
        rotation_y += 0.7
        rotation_z += 0.3
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        glRotatef(rotation_z, 0, 0, 1)
        cube()
        glPopMatrix()
        
        # Switch to 2D rendering for text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 800, 0, 600)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Render text surface
        formula_text = [
            "Rotation Formulas:",
            "x' = x",
            "y' = y·cosθ - z·sinθ",
            "z' = y·sinθ + z·cosθ",
            "",
            "Matrix Form:",
            "[x']   [1   0    0  ][x]",
            "[y'] = [0 cosθ -sinθ][y]",
            "[z']   [0 sinθ  cosθ][z]",
            "",
            f"Current Angles:",
            f"X: {rotation_x%360:.1f}°",
            f"Y: {rotation_y%360:.1f}°",
            f"Z: {rotation_z%360:.1f}°"
        ]
        
        text_surfaces = []
        for i, text in enumerate(formula_text):
            text_surface = font.render(text, True, (255, 255, 255))
            text_surfaces.append((text_surface, (20, 550 - i*20)))
        
        # Restore 3D rendering
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        
        # Blit text surfaces after flipping
        screen_surface = pygame.display.get_surface()
        for surface, pos in text_surfaces:
            screen_surface.blit(surface, pos)
        
        clock.tick(60)

if __name__ == "__main__":
    main()