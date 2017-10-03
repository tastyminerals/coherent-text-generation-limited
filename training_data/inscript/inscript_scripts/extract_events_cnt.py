#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# extract {event:cnt} map

import sys
import re
import operator
from collections import Counter

with open("inscript_events_per_topic.txt", 'r') as fopen:
    fdata = fopen.read()

rx = re.compile(r'\w+_\((\w+)\) ')
edic = Counter([w for w in rx.findall(fdata)])

rwidth = 0
for w in edic:
    width = len(w)
    if width > rwidth:
        rwidth = width

esorted = sorted(edic, key=edic.get, reverse=True)
for e in esorted:
    print e.ljust(rwidth)+str(edic[e])
