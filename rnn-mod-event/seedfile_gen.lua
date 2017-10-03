#!/usr/bin/env th
-- generates seedfile.json that is required for model sampling
-- Usage: th seedfile_gen.lua "I decided to" 30 "{1,6}" "{8,16}" bath
package.path = package.path .. ";../?.lua"

require "torch"
local JSON = require "JSON" -- luarocks install lua-json
local stringx = require "pl.stringx"

-- command line arguments
cmd = torch.CmdLine()
cmd:text("Seed generator config")
cmd:text()
cmd:text("Options:")
cmd:argument("-text","specify seed text")
cmd:argument("-sents","specify a number of sentences to generate")
cmd:argument("-events",'specify min and max number of events per sentence "{3,5}"')
cmd:argument("-words",'specify min and max number of words per sentence "{1,20}"')
cmd:argument("-topic","specify a topic name [bath, bicycle, bus, cake, flight, grocery, haircut, library, train, tree]")
cmd:text()
local opt = cmd:parse(arg or {})

local topics = {bath=1, bicycle=2, bus=3, cake=4, flight=5, grocery=6, haircut=7, library=8, train=9, tree=10}
local seedjson = {} -- seedfile table

-- set initial seed text
seedjson['initial'] = stringx.split(opt.text)

-- generate sentence miniplans
local miniplans = {}
local minmax_ev = loadstring("return "..opt.events)()
local minmax_words = loadstring("return "..opt.words)()
minmax_words = {minmax_ev[2], minmax_words[2]} -- sent length cannot be less than the max number of events
for i=1,opt.sents do
  local ev_cnt = torch.random(unpack(minmax_ev))
  local tokens_cnt = torch.random(unpack(minmax_words))
  -- increase sent length if events cnt is too high
  if ev_cnt*2 >= tokens_cnt then
    tokens_cnt = tokens_cnt + ev_cnt
  end
  miniplans[i] = topics[opt.topic]..' '..ev_cnt..' '..tokens_cnt
end
seedjson['miniplans'] = miniplans

fname = io.open("seedfile.json", "w")
io.output(fname) -- set default output file
io.write(JSON:encode_pretty(seedjson))
io.close()
