-- build and return the network
require "rnn"
require "model.GRU_mod"
require "model.GRU_mod_nodrop"


local network = {}

function network.build(traindata,opt)
  -------------------- Embeddings layer --------------------
  local net = nn.Sequential()
  local par1 = nn.ParallelTable()
  local seq_lookup = nn.Sequential()
  local lookup = nn.LookupTable(#traindata.ivocab,opt.inputsize)
  lookup.maxNorm = opt.maxnormout -- default -1
  seq_lookup:add(lookup)
  if opt.dropout > 0 then
    seq_lookup:add(nn.Dropout(opt.dropout)) -- input x
  end
  par1:add(seq_lookup)

  local seq_i1 = nn.Sequential()
  seq_i1:add(nn.Identity())
  if opt.dropout > 0 then
    seq_i1:add(nn.Dropout(opt.dropout)) -- word2vecs
  end

  local seq_i2 = nn.Sequential()
  seq_i2:add(nn.Identity())
  if opt.dropout > 0 then
    seq_i2:add(nn.Dropout(opt.dropout)) -- event vecs
  end

  local seq_i3 = nn.Sequential()
  seq_i3:add(nn.Identity())
  if opt.dropout > 0 then
    seq_i3:add(nn.Dropout(opt.dropout)) -- topic vecs
  end

  par1:add(seq_i1):add(seq_i2):add(seq_i3)
  net:add(par1)

  -- (domain adaptation) sum training and word2vec embeddings
  local concat0 = nn.ConcatTable()
  local seq0 = nn.Sequential()
  seq0:add(nn.NarrowTable(1,2)):add(nn.CAddTable())
  concat0:add(seq0)
  concat0:add(nn.NarrowTable(3,4)) -- select topic & event vectors
  net:add(concat0)

  net:add(nn.FlattenTable()) -- flatten ConcatTable output

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

return network
