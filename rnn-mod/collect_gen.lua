#!/usr/bin/env th
-- run a specified command to get a list of generated samples from the model
-- Usage: th collect_gen.lua model seedfile --temperature 0.4 --silent

package.path = package.path .. ";../?.lua"

-- command line arguments
cmd = torch.CmdLine()
cmd:text("Sampler config")
cmd:text()
cmd:text("Options:")

-- model argument, required
cmd:argument("-modelfile","specify saved model checkpoint to load the parameters from")
cmd:argument("-seedfile","specify a file containing seed text, event/topic vectors")
-- text generation options, optional
cmd:option("--seed",123,"a number used to initialize rnd generator")
cmd:option("--temperature",-1,"higher means more diversity and mistakes, only with samplemax false, [0-1] range")
cmd:option("--sents",30,"specify number of sentences to generate")
cmd:option("--silent",false,"display model call")
--cmd:option("--cuda",false,"use GPU")
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

local topics = {["TAKING A BATH"] = 1,
                ["REPAIRING A FLAT BICYCLE TIRE"] = 2,
                ["RIDING IN A PUBLIC BUS"] = 3,
                ["BAKING A CAKE"] = 4,
                ["FLYING IN AN AIRPLANE"] = 5,
                ["GOING GROCERY SHOPPING"] = 6,
                ["GETTING A HAIRCUT"] = 7,
                ["BORROWING A BOOK FROM THE LIBRARY"] = 8,
                ["GOING ON A TRAIN"] = 9,
                ["PLANTING A TREE"] = 10}

local params = ' --seed '..opt.seed..' --temperature '..opt.temperature..' --cuda'

for topic_id,seedtext in ipairs(seeds) do
  local call = 'th sampler.lua '..opt.modelfile..' '..opt.seedfile..' --seedtext "'..seedtext..'" --topic '..topic_id..params
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