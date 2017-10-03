#!/usr/bin/env th
-- display checkpoint hyperparams
require "torch"
require "rnn"

-- command line arguments
cmd = torch.CmdLine()
cmd:text("Checkpoint info config")
cmd:text()
cmd:text("Options:")

cmd:argument("-checkpoint","specify saved model checkpoint")
cmd:option("--cuda",true,"use cuda")
cmd:text()
local opt = cmd:parse(arg or {})

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(1)
end

checkpoint = torch.load(opt.checkpoint)

print(checkpoint.model)
local vocabsize = 0
for k,v in pairs(checkpoint.vocab) do vocabsize = vocabsize + 1 end
print('Dataset path:\t'..checkpoint.dataset)
print('Vocab size:\t'..vocabsize)
print('Epochs trained:\t'..checkpoint.epoch)
print('PPL (train):\t'..checkpoint.trainppl[checkpoint.epoch])
print('PPL (valid):\t'..checkpoint.valppl[checkpoint.epoch])
print('Best PPL:\t'..checkpoint.minvalppl)
