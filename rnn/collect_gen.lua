#!/usr/bin/env th
-- run a specified command to get a list of generated samples from the model
-- Usage: th collect_gen.lua model --temperature 0.4

package.path = package.path .. ";../?.lua"

-- command line arguments
cmd = torch.CmdLine()
cmd:text("Sampler config")
cmd:text()
cmd:text("Options:")

-- model argument, required
cmd:argument("-modelfile","specify saved model checkpoint to load the parameters from")
-- text generation options, optional
cmd:option("--seed",123,"a number used to initialize rnd generator")
cmd:option("--temperature",-1,"higher means more diversity and mistakes, only with samplemax false, [0-1] range")
cmd:option("--sents",30,"specify number of sentences to generate")
cmd:option("--silent",false,"display model call")
cmd:text()
local opt = cmd:parse(arg or {})

local seeds = {"I got ready to", -- TAKING A BATH
               "We planned a trip but", -- REPAIRING A FLAT BICYCLE TIRE
               "I rode the", -- RIDING IN A PUBLIC BUS
               "I warmed up the oven and", -- BAKING A CAKE
               "My wife and I were", -- FLYING IN AN AIRPLANE
               "We were all out of", -- GOING GROCERY SHOPPING
               "It was time to", -- GETTING A HAIRCUT
               "I wanted to read a new novel", -- BORROWING A BOOK FROM THE LIBRARY
               "Taking the train into the city is", -- GOING ON A TRAIN
               "I bought a flower in the shop and"} -- PLANTING A TREE

local length = opt.sents * 20 -- non seedfile models generate text using length param
local params = ' --seed '..opt.seed..' --temperature '..opt.temperature..' --length '..length

for i,seedtext in ipairs(seeds) do
  local call = 'th sampler.lua '..opt.modelfile..' --seedtext "'..seedtext..'"'..params
  if not opt.silent then
    print('Collect>>> '..call)
  end
  local pfile = io.popen(call)
  local ans = {}
  for line in pfile:lines() do
    table.insert(ans, line)
    if #ans == opt.sents then -- number of sents to generate per call
      break
    end
  end
  pfile:close()

  if opt.silent then
    for i,sent in ipairs(ans) do
      print(sent)
    end
  else
    print(ans)
  end
  print('')
end