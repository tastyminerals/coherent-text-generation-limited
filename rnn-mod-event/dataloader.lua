--[[
Collection of misc functions to work with model datasets including
preprocessing, converting to tensors, batch iteration, loading word2vec embeddings etc.
Some functions were adapted from https://github.com/Element-Research/dataload

MISC NOTES:
Each new line of your dataset must begin with a whitespace.
Ideally, your dataset should contain one sentence per line.
This is how PennTreebank preprocessed in Torch.

#How SequenceLoader works?
Load the dataset, convert it to torch.IntTensor and wrap into SequenceLoader.
SequenceLoader -- encapsulates a sequence for training time-series or language models.
The sequence is a tensor where the first dimension indexes time. Internally,
the loader will split the sequence into batchsize subsequences. Calling the
sub(start, stop, inputs, targets) method will return inputs and targets of size
(seqlen x batchsize [x inputsize]) where stop-start+1 <= seqlen.
]]

local dataloader = {}

require "classes.DataLoader"
require "classes.SequenceLoader"
require "classes.SequenceVecLoader"
require "classes.SequenceWord2VecLoader"
local path = require "pl.path"
local file = require "pl.file"
local stringx = require "pl.stringx"
local tablex = require "pl.tablex"
local cjson = require "cjson"


-- load json file, parse and convert it to table
function json2tab(fname)
  local fdata = file.read(fname)
  -- decode string keys to numbers
  local decoded = cjson.decode(fdata)
  local decoded_num = {}
  -- convert str keys to number
  for k in pairs(decoded) do decoded_num[tonumber(k)] = decoded[k] end
  -- sort
  local decoded_sorted = {}
  for i in tablex.sort(decoded_num) do table.insert(decoded_sorted, decoded_num[i]) end
  return decoded_sorted
end


-- return files from a given dir path, optionally use regex to filter the files
function listdir(path,filter)
  local i,files = 0,{}
  local pfile = io.popen('ls "'..path..'"') -- TODO: replace ls with paths
  for fname in pfile:lines() do
    if fname:match(filter or '.txt') then
      i = i + 1
      -- path must end on / or \
      fpath = path..'/'..fname
      files[i] = fpath
    end
  end
  pfile:close()
  return files
end


function max_sent_len(sents)
  local max = 0
  for i=1,#sents do
    local cnt = #stringx.split(sents[i])
    if max < cnt then max = cnt end
  end
  return max
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

function dataloader.text2tensor(tokens, vocab, set)
  local oov = vocab['<OOV>']
  local tensor = torch.IntTensor(#tokens):fill(0)
  for i, word in ipairs(tokens) do
    local wordid = vocab[word]
    if set == 'valid' and not wordid then -- we have OOV in validation, skip it or it throws CUDA device asserts
      local wordid = vocab['<unk>']
    end
    if not wordid then
      assert(oov)
      wordid = oov
    end
    -- i represents time index here and will be used by SequenceLoader
    tensor[i] = wordid
  end
  return tensor
end


function dataloader.load_data(dirname, batchsize, eos_delim, filter)
  -- the size of the batch is fixed for SequenceLoaders
  batchsize = torch.type(batchsize) == 'table' and batchsize or {batchsize, batchsize, batchsize}
  assert(torch.type(batchsize[1]) == 'number')
  local vocab
  -- load raw data, convert to tensor
  local loaders = {}
  -- get dirname files
  local files = listdir(dirname, filter)
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
    -- calculate max sent length, needed for vectors initialization
    local sents = stringx.split(data,'\n')
    local maxsent = max_sent_len(sents)
    data = stringx.replace(data, '\n', eos_delim)
    local tokens = stringx.split(data)
    print(string.format("%s, tokens: %d, max sentlen: %d",filepath,#tokens,maxsent))
    if set == "train" and not vocab then
      vocab, ivocab, wordfreq = dataloader.build_vocab(tokens)
    end
    local tensor = dataloader.text2tensor(tokens, vocab, set)
    -- encapsulate into SequenceLoader
    local loader = SequenceLoader(tensor, batchsize[i])
    loader.vocab = vocab
    loader.ivocab = ivocab
    loader.wordfreq = wordfreq
    -- additional custom fields
    loader.total_tokens = #tokens
    loader.maxsentlen = maxsent
    loader.tokens = tokens -- data split into tokens
    loader.sentscnt = table.maxn(sents) -- number of sentences
    table.insert(loaders, loader)
  end
  return unpack(loaders)
end


--[[Generates a globally unique identifier.
If a namespace is provided it is concatenated with
the time of the call, and the next value from a sequence
to get a pseudo-globally-unique name.
Otherwise, we concatenate the linux hostname]]
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


--[[ Convert vector to simple 1s vector representation.
For example, token events representation:
{
  1 : 3
  -1 : 6
  173 : 1
}
must be converted to n-dim x 1 vector: [1,1,1,1,0,0,0,...,0],
where the number of ones is equal to the total count of non zero events.
]]
function evec2simple(events,maxdim)
  local vec = torch.IntTensor(maxdim):fill(0)
  local esum = 0
  for event_id in pairs(events) do
    if event_id ~= '-1' then -- -1 is the no event id
      esum = esum + events[event_id]
    end
  end
  if esum ~= 0 then
    vec[{{1,esum}}] = 1
  end
  return vec
end


-- debugging function, returns all zeros event vector
function evec2zeros(maxdim)
  return torch.DoubleTensor(maxdim):fill(0)
end


-- convert {3:1, -1:10} vector --> [5,1,1,1,1,1,1,1,1,1,1,0...0]
function evec2full(events, weight, maxdim)
  local vec = torch.IntTensor(maxdim):fill(0)
  local esum, esum_noid = 0, 0
  for event_id in pairs(events) do
    if event_id ~= '-1' then -- -1 is the no event id
      esum = esum + events[event_id]
    else
      esum_noid = esum_noid + events[event_id]
    end
  end
  -- fill the event vector with event weight values
  if esum ~= 0 then
    vec[{{1, esum}}] = weight or 5
  end
  -- fill the event vector with no event value 1
  if esum_noid ~= 0 then
    vec[{{esum+1, esum+esum_noid}}] = 1
  end

  return vec
end


-- convert vector representation to 1-hot (not working properly)
function evec2onehot(events)
  local dim = 174 -- total max available vectors 173 and + 1 for no event vector
  local pad = math.floor(math.log(dim, 2)+1) -- zeros padding
  local onehot= torch.IntTensor(dim):fill(0)

  local function tobin(n)
    local b = {}
    for i=pad,1,-1 do
      b[i] = math.fmod(n,2)
      n = (n-b[i])/2
    end
    return tonumber(table.concat(b))
  end

  for i=1,dim do
    onehot[i] = tobin(events[tostring(i)] or 0) or 0
  end
  return onehot
end


-- load .json vectors
function dataloader.load_json_vectors(dir,evfname,topfname,batchsize,dims)
  local evfile = listdir(dir,evfname)
  local topfile = listdir(dir,topfname)
  local decoded_events = json2tab(evfile[1])
  local decoded_topics = json2tab(topfile[1])
  -- calculate #topics
  local tcnt = 0
  for _,value in ipairs(decoded_topics) do
    if tcnt < value then tcnt = value end
  end
  -- converting json data to vectors
  local vecs = {} -- token --> events vector
  local topics = {} -- token --> topic vector
  local vec -- events vector
  local topic -- topic vector
  for sent_num in pairs(decoded_events) do
    for token_num=1,#decoded_events[sent_num] do
      --vec = evec2zeros(dims[2])
      vec = evec2simple(decoded_events[sent_num][token_num],dims[2])
      --vec = evec2onehot(decoded_events[sent_num][token_num],dims[2])
      vecs[#vecs+1] = vec
      -- create a topic vector for current token
      local topic = torch.IntTensor(tcnt):fill(0)
      topic[decoded_topics[sent_num]] = 1
      topics[#topics+1] = topic
    end
  end
  -- create events IntTensor for SequenceLoader
  local tensor1 = torch.IntTensor(unpack(dims))
  for i=1,#vecs do tensor1[i] = vecs[i] end
  -- transform 15545x55x1 --> 485x32x55x1
  local events_loader = SequenceVecLoader(tensor1,batchsize)
  events_loader.events = tensor1 -- to be used in tests
  -- create topics IntTensor for SequenceLoader
  local tensor2 = torch.IntTensor(tensor1:size()[1],tcnt,tensor1:size()[3])
  for i=1,#topics do tensor2[i] = topics[i] end
  -- transform
  local topics_loader = SequenceVecLoader(tensor2,batchsize)
  topics_loader.topics = tensor2
  return events_loader, topics_loader
end


-- extract word2vec embedding for each corresponding data token and return a sequence
function dataloader.load_w2v(w2v_path,data)
  local w2v = torch.load(w2v_path)
  local batchsize = data.batchsize
  local tensor = torch.FloatTensor(#data.tokens,300)
  -- iterate through your training data and replace tokens with word2vec embeddings
  for i,token in ipairs(data.tokens) do
    if w2v.w2i[token] then
      tensor[{{i},{}}] = w2v.tensor[w2v.w2i[token]]
    else
      -- if token not found in word2vec, just insert zeros vector
      tensor[{{i},{}}] = torch.FloatTensor(300):fill(0)
    end
  end
  -- wrap the new tensor in SequenceLoader
  local loader = SequenceWord2VecLoader(tensor,data.batchsize)
  return loader
end


-- load vectors from csv files <not a csv parser!>
-- dims = {traindata.total_tokens,maxsentlen,last dim}
function dataloader.load_csv_vectors(dir,vecfile,batchsize,dims)
  local csvpath = listdir(dir,vecfile)
  local events = {}
  local topics = {}
  -- parse csv file and extract topic, events data <csv: 1st column is topic>
  for line in io.lines(csvpath[1]) do
    -- extract topic ids
    local line_tab = stringx.split(line,',')
    -- convert topics number to 1-hot vector: 3 --> [0,0,1,0,0,0,0,0,0,0]
    local tensor = torch.IntTensor(10):fill(0)
    tensor[line_tab[1]] = 1
    topics[#topics+1] = tensor
    -- extract event ids
    local events_tab = {}
    for i=2,#line_tab,2 do
      events_tab[line_tab[i]] = line_tab[i+1]
    end
    events[#events+1] = evec2full(events_tab,5,dims[2]) -- InScript dims[2] = 91
    --events[#events+1] = evec2simple(events_tab,dims[2]) -- InScript dims[2] = 91
  end
  -- construct topics tensor {#tokens x 10 x 1}
  local topics_tensor = torch.IntTensor(#topics,10,1)
  for i=1,#topics do topics_tensor[i] = topics[i] end
  -- construct events tensor {#tokens x maxsentlen x 1}
  local events_tensor = torch.IntTensor(#topics,dims[2],1)
  for i=1,#events do events_tensor[i] = events[i] end
  -- wrap topic, event tensors into SequenceLoader
  local topics_seq = SequenceVecLoader(topics_tensor,batchsize)
  local events_seq = SequenceVecLoader(events_tensor,batchsize)
  return topics_seq, events_seq
end


-- create and return word2vec table {token, embedding}
function dataloader.word2vec(w2v_path)
  local w2v = torch.load(w2v_path)
  local w2vecs = {}
  for w,i in pairs(w2v.w2i) do
    w2vecs[w] = w2v.tensor[{{i},{}}]
  end
  return w2vecs
end


return dataloader
