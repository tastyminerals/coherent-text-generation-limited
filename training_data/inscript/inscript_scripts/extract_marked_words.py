#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# extract {word:cnt} map of words that have been tagged with event descriptor

import sys
import re
import operator
from collections import Counter

with open("inscript_events_per_topic.txt", 'r') as fopen:
    fdata = fopen.read()

rx = re.compile(r'(\w+)_\(\w+\) ')
wset = Counter([w.lower() for w in rx.findall(fdata)])
#wset.sort()

rwidth = 0
for w in wset:
    width = len(w)
    if width > rwidth:
        rwidth = width
sorted(wset.keys())
#sorted(wset.items(),key=operator.itemgetter(1), reverse=True)
for k in sorted(wset.keys()):
    print k.ljust(rwidth)+str(wset[k])
