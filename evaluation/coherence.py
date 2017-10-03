#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Runs 3 similarity measures on given texts
# Requires a file with texts where each sentence is separated by \n and
# each text by an empty line.
# Use 1 (norm by sent lengths) or 0 (don't use norm) as a second arg.
# Usage: coherence.py text.txt 0

import sys
import foltz
import overlap
import wordnet_sim
import re

assert len(sys.argv) == 3, 'Not enough arguments'

stories = {}
with open(sys.argv[1], 'r') as f:
    fdata = f.read().split('\n')

# normalize by sent length
isnorm = True if sys.argv[2] == '1' else False

n = 0
stories[n] = ''
story = []
for line in fdata:
    line = re.sub(r'<punct>$', '.\n', line)
    line = re.sub(r'<punct>', ',', line)
    if re.match(r'[\t\n]+', line) or not line:
        if story:
            stories[n] = '\n'.join(story)
            n += 1
        story = []
        continue
    story.append(line.strip())

coh1, coh2, coh3, coh4 = 0, 0, 0, 0

# Barzilay & Lapata found: Ergrid, overlap, LSA, HStO and Lesk work best
# Also, Jcon and Resnik show relatively good correlation with humans.

scores = {}
for i, story in stories.items():
    print '\n\nSTORY:', i
    print '\n>>> OVERLAP'
    coh1 = overlap.coherence(story, norm=isnorm)
    print '\n>>> SENT COSSIM'
    coh2 = foltz.coherence(story, norm=isnorm)
    print '\n>>> WORDNET'
    # orig, simple, adapted, cos
    # path, lch (Leacock-Chodorow), wup (Wu-Palmer), jcn (Jiang-Conrath), res (Resnik),  lin (Lin)
    coh3 = wordnet_sim.coherence(story, opt='jcn', lesk_algorithm="adapted", norm=isnorm)
    coh4 = wordnet_sim.coherence(story, opt='res', lesk_algorithm="adapted", norm=isnorm)
    scores[i] = [coh1, coh2, coh3, coh4]

print ''
header = ['sample_id', 'overlap', 'cossim', 'jcn', 'res']
print '\t'.join(header)
coh1_sum, coh2_sum, coh3_sum, coh4_sum = 0, 0, 0, 0
for k, v in scores.items():
    coh1_sum += v[0]
    coh2_sum += v[1]
    coh3_sum += v[2]
    coh4_sum += v[3]
    print '\t'.join([str(k), str(v[0]), str(v[1]), str(v[2]), str(v[3])])

print 'Sums:\t' + '\t'.join([str(coh1_sum), str(coh2_sum),
                                      str(coh3_sum), str(coh4_sum)])

cumul = 0
for k in scores:
    cumul += sum(scores[k])
print "Cumulative sum:", cumul
