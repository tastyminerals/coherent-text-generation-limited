--[[
Data loader and other useful functions adapted from https://github.com/Element-Research/dataload

Load the dataset, convert it to torch.IntTensor and wrap into SequenceLoader.
SequenceLoader -- encapsulates a sequence for training time-series or language models.
The sequence is a tensor where the first dimension indexes time. Internally,
the loader will split the sequence into batchsize subsequences. Calling the
sub(start, stop, inputs, targets) method will return inputs and targets of size
(seqlen x batchsize [x inputsize]) where stop-start+1 <= seqlen.

Return train,valid,test sets.

Each new line of your dataset must begin with a whitespace.
Ideally, your dataset should contain one sentence per line.
This is how PennTreebank preprocessed in Torch.
<not touching the implementation for now>
]]

local dataloader = {}

require "torch"
require "rnn"
require "loader.DataLoader"
require "loader.SequenceLoader"

-- if a dir contains files, return a table of file full paths
function get_files(path)
    local i,files = 0,{}
    local pfile = io.popen('ls "'..path..'"')
    for fname in pfile:lines() do
        i = i + 1
        -- path must end on / or \
        fpath = path..'/'..fname
        files[i] = fpath
    end
    pfile:close()
    return files
end


-- this can be inefficent, I wouldn't use it for datasets larger than PTB
function dataloader.build_vocab(tokens, minfreq)
  -- setting minfreq here to let "valid"/"test" have oov words
  minfreq = 2
  assert(torch.type(tokens) == 'table', 'Expecting table')
  assert(torch.type(tokens[1]) == 'string', 'Expecting table of strings')
  minfreq = minfreq or -1
  assert(torch.type(minfreq) == 'number')
  local wordfreq = {}
  for i=1,#tokens do
    local word = tokens[i]
    wordfreq[word] = (wordfreq[word] or 0) + 1
  end
  local vocab, ivocab = {}, {}
  local wordseq = 0
  local _ = require 'moses'
  -- make sure ordering is consistent
  local words = _.sort(_.keys(wordfreq))
  local oov = 0
  for i, word in ipairs(words) do
    local freq = wordfreq[word]
    if freq >= minfreq then
      wordseq = wordseq + 1
      vocab[word] = wordseq
      ivocab[wordseq] = word
    else
      oov = oov + freq
    end
  end
  if oov > 0 then
    wordseq = wordseq + 1
    wordfreq['<OOV>'] = oov
    vocab['<OOV>'] = wordseq
    ivocab[wordseq] = '<OOV>'
  end
  return vocab, ivocab, wordfreq
end


function dataloader.text2tensor(tokens, vocab)
  local oov = vocab['<OOV>']
  local tensor = torch.IntTensor(#tokens):fill(0)
  for i, word in ipairs(tokens) do
    local wordid = vocab[word]
    if not wordid then
      assert(oov)
      wordid = oov
    end
    -- i represents time index here and will be used by SequenceLoader
    tensor[i] = wordid
  end
   return tensor
end


function dataloader.load(dirname, batchsize, path)
  -- the size of the batch is fixed for SequenceLoaders
  batchsize = torch.type(batchsize) == 'table' and batchsize or {batchsize, batchsize, batchsize}
  assert(torch.type(batchsize[1]) == 'number')
  local vocab
  -- load raw data, convert to tensor
  local file = require('pl.file')
  local stringx = require('pl.stringx')
  local loaders = {}
  -- get dirname files
  local files = get_files(dirname)
  for i,set in ipairs{'train', 'valid', 'test'} do
    local filepath
    -- find the corresponding set fname
    for _,fname in pairs(files) do
      if fname:gmatch(set)() == set then
        filepath = fname
        break
      end
    end

    if not filepath then break end
    local data = file.read(filepath)
    data = stringx.replace(data, '\n', '<eos>')
    local tokens = stringx.split(data)
    print(filepath,#tokens)
    if set == "train" and not vocab then
      vocab, ivocab, wordfreq = dataloader.build_vocab(tokens)
    end
    local tensor = dataloader.text2tensor(tokens, vocab)
    -- encapsulate into SequenceLoader
    local loader = SequenceLoader(tensor, batchsize[i])
    loader.vocab = vocab
    loader.ivocab = ivocab
    loader.wordfreq = wordfreq
    table.insert(loaders, loader)
  end
  return unpack(loaders)
end


-- Generates a globally unique identifier.
-- If a namespace is provided it is concatenated with
-- the time of the call, and the next value from a sequence
-- to get a pseudo-globally-unique name.
-- Otherwise, we concatenate the linux hostname
local counter = 1
function dataloader.uniqueid(namespace, separator)
  function hostname()
    local f = io.popen ("/bin/hostname")
    if not f then
      return 'localhost'
    end
    local hostname = f:read("*a") or ""
    f:close()
    hostname =string.gsub(hostname, "\n$", "")
    return hostname
  end

  local separator = separator or ':'
  local namespace = namespace or hostname()
  local uid = namespace..separator..os.time()..separator..counter
  counter = counter + 1
  return uid
end


return dataloader
