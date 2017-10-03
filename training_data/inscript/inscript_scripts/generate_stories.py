#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Generate incoherent stories from inscript_stories.json

import sys
import json
import random


# "good" stories are actual InScript stories (coherent stories)
# "bad" are randomly selected sents from InScript (incoherent stories)
assert len(sys.argv) > 2, 'Please provide stories cnt and type "good" or "bad"'
CNT = int(sys.argv[1])
SENTS_PER_STORY_MEDIAN = 12

with open('inscript_stories.json', 'rb') as f:
    stories = json.load(f)

if sys.argv[2] == "good":
    # generate coherent stories from InScript
    for n in xrange(CNT):
        #print '\nSTORY', n + 1
        rndtopic = random.choice(stories.keys())
        rndnum = random.choice(stories[rndtopic].keys())
        print
        print '>>>', rndtopic
        for s in stories[rndtopic][rndnum]:
            print s

elif sys.argv[2] == "bad":
    # generate incoherent stories from InScript
    for n in xrange(CNT):
        print '\nSTORY', n + 1
        for _ in xrange(SENTS_PER_STORY_MEDIAN + random.randint(-4, 4)):
            rndtopic = random.choice(stories.keys())
            rndnum = random.choice(stories[rndtopic].keys())
            rndsent = random.choice(stories[rndtopic][rndnum])
            print rndsent

