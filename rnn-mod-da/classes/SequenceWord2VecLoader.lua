local dl = {}

-- DEBUG ONLY
function px(...)
  print(...)
  os.exit()
end

local SequenceWord2VecLoader, parent = torch.class('SequenceWord2VecLoader', 'DataLoader')

function SequenceWord2VecLoader:__init(sequence, batchsize, bidirectional)
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
  local d1 = sequence:size(2) -- get the size of the second dim
  -- seqlen2 x batchsize x vecdim1
  self.data = sequence:sub(1,seqlen2*batchsize):view(batchsize,seqlen2,d1):permute(2,1,3):contiguous()
end

-- inputs : seqlen x batchsize [x inputsize]
-- targets : seqlen x batchsize [x inputsize]
function SequenceWord2VecLoader:sub(start, stop, inputs, targets)
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

function SequenceWord2VecLoader:sample()
  error"Not Implemented"
end

-- returns size of sequences
function SequenceWord2VecLoader:size()
  if self.bidirectional then
    return self.data:size(1)
  else
    return self.data:size(1)-1
  end
end

function SequenceWord2VecLoader:isize(excludedim)
  -- by default, sequence dimension is excluded
  excludedim = excludedim == nil and 1 or excludedim
  local size = torchx.recursiveSize(self.data, excludedim)
  if excludedim ~= 1 then
    size[1] = self:size()
  end
  return size
end

function SequenceWord2VecLoader:tsize(excludedim)
  return self:isize(excludedim)
end

function SequenceWord2VecLoader:subiter(seqlen, epochsize, ...)
  return parent.subiter(self, seqlen, epochsize, ...)
end

function SequenceWord2VecLoader:sampleiter(seqlen, epochsize, ...)
  error"Not Implemented. Use subiter instead."
end

return dl

