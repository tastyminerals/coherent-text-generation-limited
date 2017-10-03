------------------------------------------------------------------------
--[[ GRU ]]--
-- Author: Jin-Hwa Kim
-- License: LICENSE.2nd.txt

-- Gated Recurrent Units architecture.
-- http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
--
-- For p > 0, it becomes Bayesian GRUs [Moon et al., 2015; Gal, 2015].
-- In this case, please do not dropout on input as BGRUs handle the input with
-- its own dropouts. First, try 0.25 for p as Gal (2016) suggested, presumably,
-- because of summations of two parts in GRUs connections.
------------------------------------------------------------------------

local GRU_mod, parent = torch.class('GRU_mod', 'nn.AbstractRecurrent')

function GRU_mod:__init(opt, rho, p, mono)
  parent.__init(self, rho or 9999)
  self.p = p or 0
  if p and p ~= 0 then
    assert(nn.Dropout(p,false,false,true).lazy, 'only work with Lazy Dropout!')
  end
  self.mono = mono or false
  self.inputSize = opt.inputsize
  self.outputSize = opt.hiddensize[1]
  self.egateSize = opt.egateSize
  self.tgateSize = opt.tgateSize

  -- build the model
  self.recurrentModule = self:buildModel()
  -- make it work with nn.Container
  self.modules[1] = self.recurrentModule
  self.sharedClones[1] = self.recurrentModule

  -- for output(0), cell(0) and gradCell(T)
  self.zeroTensor = torch.Tensor()

  self.cells = {}
  self.gradCells = {}
end

-------------------------- factory methods -----------------------------
function GRU_mod:build_event_gate()
  if self.p ~= 0 then
    self.t1 = nn.Sequential()
    :add(nn.Dropout(self.p,false,false,true,self.mono))
    :add(nn.Linear(self.egateSize, self.outputSize))
    self.t2 = nn.Sequential()
    :add(nn.Dropout(self.p,false,false,true,self.mono))
    :add(nn.LinearNoBias(self.outputSize, self.outputSize))
  else
    self.t1 = nn.Linear(self.egateSize, self.outputSize)
    self.t2 = nn.LinearNoBias(self.outputSize, self.outputSize)
  end
  local para = nn.ParallelTable():add(self.t1):add(self.t2)
  local tgate = nn.Sequential()
  tgate:add(para)
  tgate:add(nn.CAddTable())
  tgate:add(nn.Sigmoid())
  return tgate
end

function GRU_mod:buildModel()
  -- input : {input, prevOutput}
  -- output : {output}
  -- Calculate four gates in one go : input, hidden, forget, output
  if self.p ~= 0 then
    self.i2g = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Dropout(self.p,false,false,true,self.mono))
      :add(nn.Dropout(self.p,false,false,true,self.mono)))
    :add(nn.ParallelTable()
      :add(nn.Linear(self.inputSize, self.outputSize))
      :add(nn.Linear(self.inputSize, self.outputSize)))
    :add(nn.JoinTable(2))

    self.o2g = nn.Sequential()
    :add(nn.ConcatTable()
      :add(nn.Dropout(self.p,false,false,true,self.mono))
      :add(nn.Dropout(self.p,false,false,true,self.mono)))
    :add(nn.ParallelTable()
      :add(nn.LinearNoBias(self.outputSize, self.outputSize))
      :add(nn.LinearNoBias(self.outputSize, self.outputSize)))
    :add(nn.JoinTable(2))
  else
    self.i2g = nn.Linear(self.inputSize, 2*self.outputSize)
    self.o2g = nn.LinearNoBias(self.outputSize, 2*self.outputSize)
  end

  local para = nn.ParallelTable():add(self.i2g):add(self.o2g)
  local gates = nn.Sequential() -- reset/update gates
  gates:add(para)
  gates:add(nn.CAddTable())
  -- reshape to (batch_size, n_gates, hid_size)
  -- then slice the n_gates dimension, i.e dimension 2
  gates:add(nn.Reshape(2,self.outputSize))
  gates:add(nn.SplitTable(1,2))
  local transfer = nn.ParallelTable()
  transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
  gates:add(transfer)

  -- input transformation
  local transform = nn.Sequential()
  transform:add(nn.FlattenTable())
  local concat = nn.ConcatTable() -- concat0
  local c1 = nn.ConcatTable()
  c1:add(nn.SelectTable(2)):add(nn.SelectTable(4))
  local c2 = nn.ConcatTable()
  c2:add(nn.SelectTable(1)):add(nn.SelectTable(4))

  concat
    :add(nn.SelectTable(1))
    :add(nn.SelectTable(3))
    :add(c1)
    :add(nn.SelectTable(4))
    :add(c2)

  transform:add(concat)

  local egate = self:build_event_gate()
  local par1 = nn.ParallelTable()
  par1
    :add(nn.Identity()) -- keep xt
    :add(nn.Identity()) -- keep events vec
    :add(egate) -- add event gate
    :add(nn.Identity()) -- keep ht-1
    :add(gates) -- compute r,z gates

  -- top Sequential() container
  local top = nn.Sequential()
  top:add(transform)
  top:add(par1)
  top:add(nn.FlattenTable()) -- {xt, events, topic, ht-1, r, z}

  -- Rearrange to {xt, events, topic, ht-1, r, z, ht-1}
  local concat = nn.ConcatTable() -- concat1
  concat:add(nn.NarrowTable(1,6)):add(nn.SelectTable(4)) -- select-add ht-1
  top:add(concat):add(nn.FlattenTable())

  -- h
  local hidden = nn.Sequential()
  local concat = nn.ConcatTable() -- concat2
  local t1 = nn.Sequential()
  t1:add(nn.SelectTable(1))  -- xt
  local t2 = nn.Sequential()
  t2:add(nn.NarrowTable(4,2)):add(nn.CMulTable()) -- ht-1,r
  local t3 = nn.Sequential()
  t3:add(nn.SelectTable(3)) -- event vector

  if self.p ~= 0 then
    t1:add(nn.Dropout(self.p,false,false,true,self.mono))
    t2:add(nn.Dropout(self.p,false,false,true,self.mono))
    t3:add(nn.Dropout(self.p,false,false,true,self.mono))
  end
  t1:add(nn.Linear(self.inputSize, self.outputSize))
  t2:add(nn.LinearNoBias(self.outputSize, self.outputSize)) -- Wh(r * ht-1)
  t3:add(nn.Linear(self.outputSize, self.outputSize)) -- Uh * event vector

  concat:add(t1):add(t2):add(t3)
  hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())

  local z1 = nn.Sequential()
  z1:add(nn.SelectTable(6))
  z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

  local o1 = nn.Sequential()
  local concat = nn.ConcatTable() -- concat3
  concat:add(hidden):add(z1)
  o1:add(concat):add(nn.CMulTable())

  local z2 = nn.Sequential()
  z2:add(nn.NarrowTable(6,2))
  z2:add(nn.CMulTable())

  local e0 = nn.Sequential()
  e0:add(nn.SelectTable(2)) -- topic vector
  if self.p ~= 0 then
    e0:add(nn.Dropout(self.p,false,false,true,self.mono))
  end
  e0:add(nn.Linear(self.tgateSize, self.outputSize)) -- St * t

  local o2 = nn.Sequential()
  local concat = nn.ConcatTable() -- concat4
  concat:add(o1):add(z2):add(e0)
  o2:add(concat):add(nn.CAddTable())

  top:add(o2)
  return top
end

------------------------- forward backward -----------------------------
function GRU_mod:updateOutput(input)
  -- input DoubleTensor batchsize x hidden
  local prevOutput
  if self.step == 1 then
    prevOutput = self.userPrevOutput or self.zeroTensor
    if input[1]:dim() == 2 then
      self.zeroTensor:resize(input[1]:size(1), self.outputSize):zero()
    else
      self.zeroTensor:resize(self.outputSize):zero()
    end
  else
    -- previous output and cell of this module
    prevOutput = self.outputs[self.step-1]
  end

  -- output(t) = GRU_mod{input(t), output(t-1)}
  local output
  if self.train ~= false then
    self:recycle()
    local recurrentModule = self:getStepModule(self.step)
    -- the actual forward propagation
    output = recurrentModule:updateOutput{input, prevOutput}
  else
    output = self.recurrentModule:updateOutput{input, prevOutput}
  end
  self.outputs[self.step] = output
  self.output = output

  self.step = self.step + 1
  self.gradPrevOutput = nil
  self.updateGradInputStep = nil
  self.accGradParametersStep = nil
  return self.output
end

function GRU_mod:_updateGradInput(input, gradOutput)
  assert(self.step > 1, "expecting at least one updateOutput")
  local step = self.updateGradInputStep - 1
  assert(step >= 1)
  local gradInput
  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  if self.gradPrevOutput then
    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
    nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
    gradOutput = self._gradOutputs[step]
  end

  local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
  local inputTable = {input, output}
  local gradInputTable = recurrentModule:updateGradInput(inputTable, gradOutput) -- failing line
  gradInput, self.gradPrevOutput = unpack(gradInputTable)
  if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end

  return gradInput
end

function GRU_mod:_accGradParameters(input, gradOutput, scale)
  local step = self.accGradParametersStep - 1
  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
  local inputTable = {input, output}
  local gradOutput = (step == self.step-1) and gradOutput or self._gradOutputs[step]
  recurrentModule:accGradParameters(inputTable, gradOutput, scale)
  return gradInput
end

function GRU_mod:__tostring__()
  return string.format('%s(%d -> %d, %.2f)', torch.type(self), self.inputSize, self.outputSize, self.p)
end

-- migrate GRU params to BGRUs params
function GRU_mod:migrate(params)
  local _params = self:parameters()
  assert(self.p ~= 0, 'only support for BGRUs.')
  assert(#params == 6, '# of source params should be 6.')
  assert(#_params == 9, '# of destination params should be 9.')
  _params[1]:copy(params[1]:narrow(1,1,self.outputSize))
  _params[2]:copy(params[2]:narrow(1,1,self.outputSize))
  _params[3]:copy(params[1]:narrow(1,self.outputSize+1,self.outputSize))
  _params[4]:copy(params[2]:narrow(1,self.outputSize+1,self.outputSize))
  _params[5]:copy(params[3]:narrow(1,1,self.outputSize))
  _params[6]:copy(params[3]:narrow(1,self.outputSize+1,self.outputSize))
  _params[7]:copy(params[4])
  _params[8]:copy(params[5])
  _params[9]:copy(params[6])
end
