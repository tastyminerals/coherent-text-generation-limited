local dl = {}

-- DEBUG ONLY
function px(...)
  print(...)
  os.exit()
end

local SequenceVecLoader, parent = torch.class('SequenceVecLoader', 'DataLoader')

function SequenceVecLoader:__init(sequence, batchsize, bidirectional)
  assert(torch.isTensor(sequence))
  assert(torch.type(batchsize) == 'number')
  -- sequence is a tensor where the first dimension indexes time
  self.batchsize = batchsize
  self.bidirectional = bidirectional
  -- itersize field is used for parallel iteration (moses/penlight zip functions won't work here)
  self.itersize = torch.floor(sequence:size():totable()[1] / batchsize)
  local seqlen = sequence:size(1)
  local size = sequence:size():totable()
  table.remove(size, 1)
  assert(#size == sequence:dim() - 1)
  self.data = sequence.new()
  -- note that some data will be lost
  local seqlen2 = torch.floor(seqlen / batchsize)
  -- get single vector dimensions from sequence 15000x54x1, e.g. 54x1
  local d1,d2 = sequence:size(2),sequence:size(3)
  -- seqlen2 x batchsize x vecdim1 x vecdim2
  self.data = sequence:sub(1,seqlen2*batchsize):view(batchsize,seqlen2,d1,d2):permute(2,1,3,4):contiguous()
end

-- inputs : seqlen x batchsize [x inputsize]
-- targets : seqlen x batchsize [x inputsize]
function SequenceVecLoader:sub(start, stop, inputs, targets)
  local seqlen = stop - start + 1

  inputs = inputs or self.data.new()
  targets = targets or inputs.new()

  if self.bidirectional then
    assert(stop <= self.data:size(1))
    inputs:set(self.data:sub(start, stop))
    targets:set(inputs)
  else
    assert(stop < self.data:size(1))
    inputs:set(self.data:sub(start, stop))
    targets:set(self.data:sub(start+1, stop+1))
  end

  return inputs, targets
end

function SequenceVecLoader:sample()
  error"Not Implemented"
end

-- returns size of sequences
function SequenceVecLoader:size()
  if self.bidirectional then
    return self.data:size(1)
  else
    return self.data:size(1)-1
  end
end

function SequenceVecLoader:isize(excludedim)
  -- by default, sequence dimension is excluded
  excludedim = excludedim == nil and 1 or excludedim
  local size = torchx.recursiveSize(self.data, excludedim)
  if excludedim ~= 1 then
    size[1] = self:size()
  end
  return size
end

function SequenceVecLoader:tsize(excludedim)
  return self:isize(excludedim)
end

function SequenceVecLoader:subiter(seqlen, epochsize, ...)
  return parent.subiter(self, seqlen, epochsize, ...)
end

function SequenceVecLoader:sampleiter(seqlen, epochsize, ...)
  error"Not Implemented. Use subiter instead."
end

return dl

