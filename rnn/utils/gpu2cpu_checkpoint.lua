--[[
A quick script for converting GPU checkpoints to CPU checkpoints.
CPU checkpoints are not saved by the training script automatically
because of Torch cloning limitations. In particular, it is not
possible to clone a GPU model on CPU, something like :clone():float()
with a single call, without needing extra memory on the GPU. If this
existed then it would be possible to do this inside the training
script without worrying about blowing up the memory.
]]--

require 'torch'
require 'rnn'
require 'nngraph'
require 'cutorch'
require 'cunn'
--require 'cudnn' -- only needed if the loaded model used cudnn as backend. otherwise can be commented out
-- local imports
require "loader.DataLoader"
require "loader.SequenceLoader"


cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a GPU checkpoint to CPU checkpoint.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','GPU model checkpoint to convert')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.DoubleTensor') -- for CPU
--torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed

local checkpoint = torch.load(opt.model)

-------------------------------------------------------------------------------
function ship2cpu(vals)
  return vals:double()
  --return vals:float()
end
-------------------------------------------------------------------------------

for k,vals in pairs(checkpoint) do
  if k == "criterion" then
    -- enter criterion and :copy()/:float() everything
    checkpoint[k] = ship2cpu(vals)
  elseif k == "model" then
    -- enter model and :copy()/:float() everything
    checkpoint[k] = ship2cpu(vals)
  elseif k == "targetmodule" then
    -- enter model and :copy()/:float() everything
    checkpoint[k] = ship2cpu(vals)
  end
end

local savefile = opt.model .. '_cpu.t7' -- append "cpu.t7" to filename
torch.save(savefile, checkpoint)
print('saved ' .. savefile)
