require "rnn"
require "model.GRU_mod"
require "model.GRU_mod_nodrop"

local network2 = {}

function network2.build(traindata,opt)
  -------------------- Embeddings layer --------------------
  local net = nn.Sequential()
  local par1 = nn.ParallelTable()
  local seq_i0 = nn.Sequential()
  local lookup = nn.LookupTable(#traindata.ivocab,opt.inputsize) -- (17134x300)
  lookup.maxNorm = -1

  seq_i0:add(lookup)

  if opt.dropout > 0 then
    seq_i0:add(nn.Dropout(opt.dropout))
  end

  par1:add(seq_i0)

  local seq_i1 = nn.Sequential()
  seq_i1:add(nn.Identity())
  local seq_i2 = nn.Sequential()
  seq_i2:add(nn.Identity())

  if opt.dropout > 0 then
    seq_i1:add(nn.Dropout(opt.dropout))
    seq_i2:add(nn.Dropout(opt.dropout))
  end

  par1:add(seq_i1):add(seq_i2)
  --par1:add(nn.Identity()):add(nn.LookupTable(traindata.network1_ivocab,opt.inputsize))
  net:add(par1)
  net:add(nn.MapTable():add(nn.SplitTable(1)))
  net:add(nn.ZipTable())


  -------------------- Recurrent layer --------------------
  local stepmodule = nn.Sequential()
  local rnn = GRU_mod(opt, nil, opt.dropout/2)
  --local rnn = GRU_mod_nodrop(opt,nil)
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

return network2
