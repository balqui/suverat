""" 
Author: Jose L Balcazar, ORCID 0000-0003-4248-4528
License: MIT 

Feb 2017 onwards

2D SVM toy based on Balcazar/Dai/Tanaka/Watanabe 
DOI: 10.1007/s00224-007-9094-6

2D SVM on linearly separable datasets -- separability 
not checked as of today

Based on 011 version among previous versions since 2012 
in Dropbox (or lost). This and 011 version: refactored, 
migrated to Python3; this version: uploaded through 
Mercurial to Bitbucket

algorithm:
    picks up two points per color and solves
      for the best of the 4 hyperplanes
    doubles probability of mistakes under the 
      condition that total mass does not increase
      more than 25% (or should it be 16% =~ 1/6?)
      CONDITION NOT YET IMPLEMENTED!!!
    repeats until convergence

algorithm at previous version 011:
    picks up to two or up to three points per color and solves
      for the best of the up to 4 or 6 hyperplanes
    doubles probability of mistakes
    repeats until convergence

local support vectors marked in white
global support vectors marked in black at convergence

can initialize internally, guaranteeing separability

ToDo:
- control mass increase
- display splash and help blits
- additional key to ask for reporting about what happened
- stage-wise progress bar (distinguishing unlucky ones)
- check for separability
- loop on k-barycenters until separability
- add kernels (how does separability work then?)
- exe file for Windows, screensaver option on various systems
- rethink keys in a context of screensaver
- tried additional key to toggle full-screen-mode on/off
   but seems complicated to make sure it works well - postponed
- further config issues (show comb point?)
"""

from random import randrange
from math import sqrt

from points import Points
from dot import dot

from colors import black, white, red, blue, reddish, blueish

## Additional functions for handling vectors and halfspaces and screen

def pair_up(p,q,r):
    "returns the point in segment q-p that is closest to r; accepts p==q"
    segment = (q[0] - p[0], q[1] - p[1])
    segsq = dot(segment,segment)
    if segsq == 0:
        "p==q"
        return p, 0
    vnew = (r[0] - p[0], r[1] - p[1])
    alpha = dot(vnew,segment)*1.0 / segsq # we know segsq != 0
    if alpha < 0: alpha = 0
    if alpha > 1: alpha = 1
    return (alpha*q[0] + (1-alpha)*p[0], alpha*q[1] + (1-alpha)*p[1]), alpha

## draw hyperplane through point p with normal vector nv
def draw_hplane(p,nv,pc,nc):
    "pc: color for pos side (single pt), nc: color for neg side (segment)"
    for i in range(width):
        for j in range(height):
            if dot((i-p[0],j-p[1]), nv) >= 0:
                "positive halfspace, including hyperplane itself"
                myspace[i][j] = pc
            else:
                myspace[i][j] = nc

def draw_point(color, center, radius, solid=True):
    if solid:
        pygame.draw.circle(screen, color, center, radius)
    else:
        pygame.draw.circle(screen, color, center, radius, 2)

# as in pygame.draw.circle(screen,color,self.coords[i],int(size+0.1),2)

def rand_point(w,h):
    """check docs about screen size"""
    return (randrange(1,w+1),randrange(1,h+1))

def rand_hplane(w,h):
    w = w/4
    h = h/4
    pp = (randrange(w,3*w),randrange(h,3*h)) # chosen in the central area
    vv = (randrange(-w,w),randrange(-h,h))
    v_len = sqrt(dot(vv,vv))
    vv = (vv[0]/v_len,vv[1]/v_len)
    return pp, vv

def set_screen(full_screen):
    if full_screen:
        return pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    else:
        "smaller screen of hardwired size"
        return pygame.display.set_mode((601, 401))

## The real meat of the thing: the randomized training algorithm

def best_margin(ng_pts,s_n_ids,ps_pts,s_p_ids):
    "simple version for ns = 2 >= len(s_n_ids) = len(s_p_ids)"
    print "len(s_p_ids), len(s_n_ids)", len(s_p_ids), len(s_n_ids)
    m_sq_best = float("inf")
    p1 = ps_pts.coords[s_p_ids[0]]
    p2 = ps_pts.coords[s_p_ids[-1]] # [1] if len = 2, [0] if len = 1
    n1 = ng_pts.coords[s_n_ids[0]]
    n2 = ng_pts.coords[s_n_ids[-1]]

    c_p, alpha = pair_up(p1,p2,n1)
# v must point from neg to pos
    v = (c_p[0] - n1[0], c_p[1] - n1[1])
    m_sq = dot(v,v)
    if m_sq < m_sq_best:
        "better (squared) margin found"
        m_sq_best = m_sq
        v_best = v
        c_p_best = c_p
        ref = (0.5*c_p[0] + 0.5*n1[0], 0.5*c_p[1] + 0.5*n1[1])
        s_v_p = [ ]
        if alpha > 0:
            s_v_p.append(s_p_ids[1])
        if alpha < 1:
            s_v_p.append(s_p_ids[0])
        s_v_n = [ s_n_ids[0] ]
    
    c_p, alpha = pair_up(p1,p2,n2)
# three more likewise cases
    v = (c_p[0] - n2[0], c_p[1] - n2[1])
    m_sq = dot(v,v)
    if m_sq < m_sq_best:
        "better (squared) margin found"
        m_sq_best = m_sq
        v_best = v
        c_p_best = c_p
        ref = (0.5*c_p[0] + 0.5*n2[0], 0.5*c_p[1] + 0.5*n2[1])
        s_v_p = [ ]
        if alpha > 0:
            s_v_p.append(s_p_ids[1])
        if alpha < 1:
            s_v_p.append(s_p_ids[0])
        s_v_n = [ s_n_ids[1] ]
    
    c_p, alpha = pair_up(n1,n2,p1)
# v must point from neg to pos
    v = (p1[0] - c_p[0], p1[1] - c_p[1])
    m_sq = dot(v,v)
    if m_sq < m_sq_best:
        "better (squared) margin found"
        m_sq_best = m_sq
        v_best = v
        c_p_best = c_p
        ref = (0.5*c_p[0] + 0.5*p1[0], 0.5*c_p[1] + 0.5*p1[1])
        s_v_n = [ ]
        if alpha > 0:
            s_v_n.append(s_n_ids[1])
        if alpha < 1:
            s_v_n.append(s_n_ids[0])
        s_v_p = [ s_p_ids[0] ]
    
    c_p, alpha = pair_up(n1,n2,p2)
# last case
    v = (p2[0] - c_p[0], p2[1] - c_p[1])
    m_sq = dot(v,v)
    if m_sq < m_sq_best:
        "better (squared) margin found"
        m_sq_best = m_sq
        v_best = v
        c_p_best = c_p
        ref = (0.5*c_p[0] + 0.5*p2[0], 0.5*c_p[1] + 0.5*p2[1])
        s_v_n = [ ]
        if alpha > 0:
            s_v_n.append(s_n_ids[1])
        if alpha < 1:
            s_v_n.append(s_n_ids[0])
        s_v_p = [ s_p_ids[1] ]

    print "len(s_v_p), len(s_v_n):", len(s_v_p), len(s_v_n)
    return ref, v_best, c_p_best, s_v_p, s_v_n

def run_it(ng_pts, ps_pts, it):
    "go once through the loop, return true if must go on"
    ns = 2 # sampling two points per color in this version
    s_n_ids = ng_pts.sampl(ns)
    s_p_ids = ps_pts.sampl(ns)
    ref, vdir, comb_point, s_v_p, s_v_n = best_margin(ng_pts,s_n_ids,ps_pts,s_p_ids)
    draw_hplane(ref,vdir,ps_pts.color_back,ng_pts.color_back)
    ng_pts.draw_points(draw_point)
    ps_pts.draw_points(draw_point)
    w = ng_pts.test_points(ref,vdir)
    w += ps_pts.test_points(ref,vdir) ## w: how many misclassifications
    if show_comb_point:
        comb_point = int(comb_point[0]), int(comb_point[1])
        draw_point(white, comb_point, ps_pts.rad)
    if w == 0:
        ng_pts.mark_points(draw_point,s_v_n,black)
        ps_pts.mark_points(draw_point,s_v_p,black)
    else:
        ng_pts.mark_points(draw_point,s_n_ids)
        ps_pts.mark_points(draw_point,s_p_ids)
    pygame.display.flip()
    return w > 0



## start

import pygame

pygame.init()

clock = pygame.time.Clock()

show_comb_point = True # show the convex comb point that marks normal vector

full_screen = True # set it to false for smaller test screen
screen = set_screen(full_screen)
width = screen.get_width()
height = screen.get_height()
centerscreen = int(width/2), int(height/2)

neg_color = screen.map_rgb(red)
pos_color = screen.map_rgb(blue)

neg_color_back = screen.map_rgb(reddish)
pos_color_back = screen.map_rgb(blueish)

#rad = 3 # minimal point radius

myspace = pygame.surfarray.pixels2d(screen)

screen.fill(black)
#screen.blit() # the splash with the help info

pos_points = Points(pos_color,pos_color_back,+1)
neg_points = Points(neg_color,neg_color_back,-1)

points_gen = 25 # quantity of points to be generated upon hitting "g"

it = 0
running = False # to run all the way to convergence
once = False    # to run step by step
showing_help = False

while True:
    clock.tick(10)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            "mouse buttons set points"
            if event.button == 1:
                pos_points.appnd(event.pos)
                pos_points.draw_points(draw_point, [pos_points.cnt-1])
                pygame.display.flip()
            elif event.button == 3:
                neg_points.appnd(event.pos)
                neg_points.draw_points(draw_point, [neg_points.cnt-1])
                pygame.display.flip()
        if event.type == pygame.KEYDOWN:
            if event.key == 113:
                "q: quit / quit help"
                if showing_help:
                    screen.fill(black)
                    pos_points.draw_points(draw_point)
                    neg_points.draw_points(draw_point)
                    pygame.display.flip()
                    showing_help = False
                else:
                    pygame.display.quit()
                    exit()
            elif event.key == 98:
                "b: back, restart"
                it = 0
                screen.fill(black)
                pos_points.restart()
                neg_points.restart()
                pos_points.draw_points(draw_point)
                neg_points.draw_points(draw_point)
                pygame.display.flip()
            elif event.key == 99:
                "c: clear"
                it = 0
                screen.fill(black)
                pos_points.clear()
                neg_points.clear()
                pygame.display.flip()
            elif event.key == 103:
                "g: generate"
                it = 0
                screen.fill(black)
                pos_points.clear()
                neg_points.clear()
                pp, vv = rand_hplane(width,height)
                for i in range(points_gen):
                    ppp = rand_point(width,height)
                    if dot((ppp[0]-pp[0],ppp[1]-pp[1]), vv) >= 0:
                        pos_points.appnd(ppp)
                    else:
                        neg_points.appnd(ppp)
                pos_points.draw_points(draw_point)
                neg_points.draw_points(draw_point)
                pygame.display.flip()
            elif event.key == 104:
                "h: display help"
                showing_help = True
                running = False
                once = False
                # screen.blit() # display help
            elif event.key == 110 or event.key == 111:
                "n, o: run once"
                once = True
            elif event.key == 114:
                "r: run, start iterating"
                running = True
            elif event.key == 115:
                "s: stop iterating"
                running = False
                it = 0
    if running or once:
        "run one more loop"
        once = False
        it += 1
        running = run_it(neg_points,pos_points,it) and running



