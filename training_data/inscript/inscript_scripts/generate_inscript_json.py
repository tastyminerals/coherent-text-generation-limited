#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Generate json file from inscript_stories.txt file
# Calculate average token counts per sent and per story.

from __future__ import division
import os
import json
from collections import defaultdict as dd
from string import punctuation as punct


with open('inscript_stories.txt', 'r') as f:
    fdata = f.read()

stories = dd(dict)
tok_per_sent = []
tok_per_story = []
for block in fdata.split('-'*10):
    if not block: continue
    for line in block.split('\n'):
        if not line: continue
        if line.startswith('InScript/'):
            story = []
            topic, n = os.path.basename(os.path.splitext(line)[0]).split('_')
            stories[topic][n] = ''
            continue
        story.append(line)
        tok_per_sent.append(len([t for t in line.split() if t not in punct]))
    stories[topic][n] = story
    tok_per_story.append(len([t for s in story for t in s.split()
                         if t not in punct]))

with open('incoherent_stories.json', 'w') as f:
    json.dump(stories, f, indent=4, separators=(',', ': '))

total_stories = len([n for k in stories for n in stories[k]])
sent_per_story = [len(stories[k][n]) for k in stories for n in stories[k]]

print 'Average tokens per sent: %f' % (sum(tok_per_sent) / len(tok_per_sent))
print 'Average tokens per story: %f' % (sum(tok_per_story) / len(tok_per_story))
print 'Average sents per story: %f' % (sum(sent_per_story) / total_stories)
