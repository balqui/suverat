""" 
Author: Jose L Balcazar, ORCID (feb 2017)
License: MIT 

2D SVM toy, auxiliary class Points
"""

from collections import defaultdict
from random import randrange
from dot import dot, violates_margin, white

class Points:
    """
    Stores a bunch of points by coordinates plus 
    respective probabilities and one label to rule them all
    """

    def __init__(self, color, color_back, label, radius=3):
        self.clear()
        self.color = color
        self.color_back = color_back
        self.label = label
        self.rad = radius

    def clear(self):
        self.coords = {}
        self.cnt = 0
        self.prob = {}
        self.log_prob = {} ## logs of previous
        self.mass = 0

    def restart(self):
        self.prob = {}
        self.log_prob = {} ## logs of previous
        self.mass = 0
        for p in self.coords:
            self.prob[p] = 1
            self.log_prob[p] = 0 ## logs of previous
            self.mass += 1

    def appnd(self,xy):
        "len(xy) must be 2"
        self.coords[self.cnt] = xy
        self.prob[self.cnt] = 1
        self.log_prob[self.cnt] = 0
        self.mass += 1
        self.cnt += 1

    def doub(self,p):
        "add tests that p in coords ans prob"
        self.mass += self.prob[p]
        self.prob[p] += self.prob[p]
        self.log_prob[p] += 1

    def sampl(self, k):
        "sample k points according to probs"
        rprob = defaultdict(int)
        for i in range(k):
            toss = randrange(self.mass)
            j = 0
            while toss >= self.prob[j]:
                toss -= self.prob[j]
                j += 1
            rprob[j] += 1
            self.prob[j] -= 1
            self.mass -= 1
        r = []
        for j in rprob:
            r.append(j)
            self.prob[j] += rprob[j]
            self.mass += rprob[j]
        return r

    def draw_points(self, draw_point, which=[], color=None): # rd = None, removed
        if not which:
            which = range(self.cnt)
        if not color:
            color = self.color
        for i in which:
#            if rd:
#                size = rd
#            else:
            size = self.rad + self.log_prob[i]
            draw_point(color, self.coords[i], int(size + 0.1)) # why +0.1?

    def mark_points(self, draw_point, which=[], color=None): # rad removed
        if not which:
            which = range(self.cnt)
        if not color:
            color = white
        for i in which:
            size = 2*self.rad + self.log_prob[i]
            draw_point(color, self.coords[i], int(size+0.1), False) # not solid

    def test_points(self, p, nv):
        "count points outside margin and accumulate total amount of prob growth"
        wrongs = 0
        incr = 0.0
        if self.label > 0:
            "will double if neg or if pos but less than margin"
            for i in range(self.cnt):
                if violates_margin(p, nv, self.coords[i], +1):
                    incr += self.prob[i]
                    wrongs += 1
        else:
            "will double if pos or if neg but abs val less than margin"
            for i in range(self.cnt):
                if violates_margin(p, nv, self.coords[i], -1):
                    incr += self.prob[i]
                    wrongs += 1
        return wrongs, self.mass, incr

    def update_points(self, p, nv):
        "stage was good: actually change the weights"
        if self.label > 0:
            "double if neg or if pos but less than margin"
            for i in range(self.cnt):
                if violates_margin(p, nv, self.coords[i], +1):
                    self.doub(i)
        else:
            "double if pos or neg but abs val less than margin"
            for i in range(self.cnt):
                if violates_margin(p, nv, self.coords[i], -1):
                    self.doub(i)
        return

    def dump(self):
        print "Count:", self.cnt
        print "Total mass:", self.mass
        for i in range(self.cnt):
            print "(", self.coords[i][0], ",",
            print self.coords[i][1], ")  :", self.prob[i]
