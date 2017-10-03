require 'rnn'
require 'model.Sequential'
require 'model.ConcatTable'

function buildModel()
  inputSize = 100
  outputSize = 100
  p = 0
  -- input : {input, prevOutput}
  -- output : {output}
  -- Calculate all four gates in one go : input, hidden, forget, output
  if p ~= 0 then
    i2g = nn.Sequential()
    :add(ConcatTable()
      :add(nn.Dropout(p,false,false,true,mono))
      :add(nn.Dropout(p,false,false,true,mono)))
    :add(nn.ParallelTable()
      :add(nn.Linear(inputSize, outputSize))
      :add(nn.Linear(inputSize, outputSize)))
    :add(nn.JoinTable(2))
    o2g = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Dropout(p,false,false,true,mono))
      :add(nn.Dropout(p,false,false,true,mono)))
    :add(nn.ParallelTable()
      :add(nn.LinearNoBias(outputSize, outputSize))
      :add(nn.LinearNoBias(outputSize, outputSize)))
    :add(nn.JoinTable(2))
  else
    i2g = nn.Linear(inputSize, 2*outputSize)
    o2g = nn.LinearNoBias(outputSize, 2*outputSize)
  end

  local para = nn.ParallelTable():add(i2g):add(o2g)
  local gates = Sequential()
  gates:add(para)
  gates:add(nn.CAddTable())

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  gates:add(nn.Reshape(2,outputSize))
  gates:add(nn.SplitTable(1,2))
  local transfer = nn.ParallelTable()
  transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
  gates:add(transfer)

  local concat = ConcatTable():add(nn.Identity()):add(gates)
  local seq = nn.Sequential()
  seq:add(concat)
  seq:add(nn.FlattenTable()) -- x(t), s(t-1), r, z

  -- Rearrange to x(t), s(t-1), r, z, s(t-1)
  local concat = nn.ConcatTable()  --
  concat:add(nn.NarrowTable(1,4)):add(nn.SelectTable(2))
  seq:add(concat):add(nn.FlattenTable())

  -- h
  local hidden = nn.Sequential()
  local concat = nn.ConcatTable()
  local t1 = nn.Sequential()
  t1:add(nn.SelectTable(1))
  local t2 = nn.Sequential()
  t2:add(nn.NarrowTable(2,2)):add(nn.CMulTable())
  if p ~= 0 then
    t1:add(nn.Dropout(p,false,false,true,mono))
    t2:add(nn.Dropout(p,false,false,true,mono))
  end
  t1:add(nn.Linear(inputSize, outputSize))
  t2:add(nn.LinearNoBias(outputSize, outputSize))

  concat:add(t1):add(t2)
  hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())

  local z1 = nn.Sequential()
  z1:add(nn.SelectTable(4))
  z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

  local z2 = nn.Sequential()
  z2:add(nn.NarrowTable(4,2))
  z2:add(nn.CMulTable())

  local o1 = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(hidden):add(z1)
  o1:add(concat):add(nn.CMulTable())

  local o2 = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(o1):add(z2)
  o2:add(concat):add(nn.CAddTable())

  seq:add(o2)
  return seq
end

model = buildModel()
print(model)