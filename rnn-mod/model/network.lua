-- build and return the network
require "rnn"
require "model.GRU_mod"

local network = {}

function network.build(traindata,opt)

  -------------------- Embeddings layer --------------------
  local net = nn.Sequential()
  local par1 = nn.ParallelTable()
  local seq1 = nn.Sequential()
  local lookup = nn.LookupTable(#traindata.ivocab,opt.inputsize)
  lookup.maxNorm = -1
  seq1:add(lookup)
  if opt.dropout > 0 then
    seq1:add(nn.Dropout(opt.dropout))
  end
  par1:add(seq1)
  par1:add(nn.Identity()):add(nn.Identity())
  net:add(par1)
  net:add(nn.MapTable():add(nn.SplitTable(1)))
  net:add(nn.ZipTable())

  -------------------- Recurrent layer --------------------
  local stepmodule = nn.Sequential()
  local rnn = GRU_mod(opt,nil,opt.dropout/2)
  stepmodule:add(rnn)

  -------------------- LogSoftMax layer --------------------
  if opt.dropout > 0 then
    stepmodule:add(nn.Dropout(opt.dropout))
  end
  local lin = nn.Linear(opt.hiddensize[1],#traindata.ivocab)
  stepmodule:add(lin)
  stepmodule:add(nn.LogSoftMax())

  -- adding recurrency
  net:add(nn.Sequencer(stepmodule))

  -- remember previous state between batches
  net:remember((opt.rnn and "eval") or "both")
  return net
end

return network
