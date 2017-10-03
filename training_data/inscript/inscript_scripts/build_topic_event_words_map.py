#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Build a map {topic: {event: [words]}} from InScript corpus

import re
import json


with open('inscript_events_per_topic.txt', 'r') as f:
    fdata = f.read()

fspl = fdata.split('----------')
rx = re.compile(r'(\w+_\(\w+\)) ')
rxw = re.compile(r'(\w+)_\(\w+\)')
rxe = re.compile(r'\w+_\((\w+)\)')


inscript_map = {}
for block in fspl:
    if not block: continue
    blockspl = block.split('\n')
    for line in blockspl:
        if not line: continue
        if line.endswith('.xml'):
            topic = line.split('/')[2]
            inscript_map[topic] = {}
        marked = rx.findall(line)
        for mword in marked:
            word = rxw.match(mword).group(1)
            event = rxe.match(mword).group(1)
            if inscript_map[topic].get(event):
                if word not in inscript_map[topic][event]:
                    inscript_map[topic][event].append(word)
            else:
                inscript_map[topic][event] = []
                if word not in inscript_map[topic][event]:
                    inscript_map[topic][event].append(word)

"""
for topic, event_dict in inscript_map.items():
    print topic
    for event, words in event_dict.items():
        print '  ' + event
        for w in words:
            print '  '*2 + w
        print
    print '-' * 10
"""

with open("topic_event_words_map.json", "w") as fjson:
    json.dump(inscript_map, fjson, indent=2, separators=(',', ': '))

print 'topic_event_words_map.json generated'
