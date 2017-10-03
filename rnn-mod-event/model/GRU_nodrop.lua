------------------------------------------------------------------------
--[[ GRU ]]--
-- Author: Jin-Hwa Kim
-- License: LICENSE.2nd.txt

-- Gated Recurrent Units architecture.
-- http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
--
-- a version without Dropout
------------------------------------------------------------------------
local GRU_nodrop, parent = torch.class('GRU_nodrop', 'nn.AbstractRecurrent')

function GRU_nodrop:__init(inputSize, outputSize, rho, p, mono)
  parent.__init(self, rho or 9999)
  self.p = p or 0
  self.mono = mono or false
  self.inputSize = inputSize
  self.outputSize = outputSize
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
function GRU_nodrop:buildModel()
  -- input : {input, prevOutput}
  -- output : {output}

  -- Calculate all four gates in one go : input, hidden, forget, output
  self.i2g = nn.Linear(self.inputSize, 2*self.outputSize)
  self.o2g = nn.LinearNoBias(self.outputSize, 2*self.outputSize)
  local para = nn.ParallelTable():add(self.i2g):add(self.o2g)
  local gates = nn.Sequential()
  gates:add(para)
  gates:add(nn.CAddTable())

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  gates:add(nn.Reshape(2,self.outputSize))
  gates:add(nn.SplitTable(1,2))
  local transfer = nn.ParallelTable()
  transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
  gates:add(transfer)

  local concat = nn.ConcatTable():add(nn.Identity()):add(gates)
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
  t1:add(nn.Linear(self.inputSize, self.outputSize))
  t2:add(nn.LinearNoBias(self.outputSize, self.outputSize))

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

function GRU_nodrop:getHiddenState(step, input)
  local prevOutput
  if step == 0 then
    prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
    if input then
      if input:dim() == 2 then
        self.zeroTensor:resize(input:size(1), self.outputSize):zero()
      else
        self.zeroTensor:resize(self.outputSize):zero()
      end
    end
  else
    -- previous output and cell of this module
    prevOutput = self.outputs[step]
  end
  return prevOutput
end


function GRU_nodrop:setHiddenState(step, hiddenState)
  assert(torch.isTensor(hiddenState))
  self.outputs[step] = hiddenState
end

------------------------- forward backward -----------------------------
function GRU_nodrop:updateOutput(input)
  local prevOutput = self:getHiddenState(self.step-1, input)
  -- output(t) = gru{input(t), output(t-1)}
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
  -- note that we don't return the cell, just the output
  return self.output
end


function GRU_nodrop:getGradHiddenState(step)
  local gradOutput
  if step == self.step-1 then
    gradOutput = self.userNextGradOutput or self.gradOutputs[step] or self.zeroTensor
  else
    gradOutput = self.gradOutputs[step]
  end
  return gradOutput
end

function GRU_nodrop:setGradHiddenState(step, gradHiddenState)
  assert(torch.isTensor(gradHiddenState))
  self.gradOutputs[step] = gradHiddenState
end

function GRU_nodrop:_updateGradInput(input, gradOutput)
  assert(self.step > 1, "expecting at least one updateOutput")
  local step = self.updateGradInputStep - 1
  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  local _gradOutput = self:getGradHiddenState(step)
  assert(_gradOutput)
  self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
  nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
  gradOutput = self._gradOutputs[step]

  local gradInputTable = recurrentModule:updateGradInput({input, self:getHiddenState(step-1)}, gradOutput)

  self:setGradHiddenState(step-1, gradInputTable[2])

  return gradInputTable[1]
end

function GRU_nodrop:_accGradParameters(input, gradOutput, scale)
  local step = self.accGradParametersStep - 1
  assert(step >= 1)

  -- set the output/gradOutput states of current Module
  local recurrentModule = self:getStepModule(step)

  -- backward propagate through this step
  local gradOutput = self._gradOutputs[step] or self:getGradHiddenState(step)
  recurrentModule:accGradParameters({input, self:getHiddenState(step-1)}, gradOutput, scale)
end

function GRU_nodrop:__tostring__()
  return string.format('%s(%d -> %d, %.2f)', torch.type(self), self.inputSize, self.outputSize, self.p)
end

-- migrate GRUs params to BGRUs params
function GRU_nodrop:migrate(params)
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
