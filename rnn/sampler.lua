--[[
Generate text given a trained language model.

Based on: https://github.com/oxford-cs-ml-2015/practical6
and: https://github.com/larspars/word-rnn

Requires:
  trained model .t7 file
]]

require "torch"
require "rnn"
local loader = require "loader.dataloader"
local txt = require "utils.tastytxt"
local adddim = nn.utils.addSingletonDimension

-- DEBUG ONLY
function px(...)
  print(...)
  os.exit()
end

-- command line arguments
cmd = torch.CmdLine()
cmd:text("Sampler config")
cmd:text()
cmd:text("Options:")

-- model argument, required
cmd:argument("-modelfile","specify saved model checkpoint to load the parameters from")

-- text generation options, optional
cmd:option("--seed",123,"a number used to initialize rnd generator")
cmd:option("--seedtext","once upon a time","a text sample used as input to RNN before the actual generation")
cmd:option("--length",100,"number of words to generate")
cmd:option("--temperature",-1,"higher means more diversity and mistakes, only with samplemax false, [0-1] range")
cmd:option("--skipunk",false,"skip <unk> words during generation")
cmd:option("--cuda",false,"use cuda")
cmd:text()

-- init accumulated args
local opt = cmd:parse(arg or {})

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(1)
end


-- handle command line args
local seedtext = txt.tokenize(opt.seedtext:lower())
-- set rnd seed
torch.manualSeed(opt.seed)
-- load saved model checkpoint
checkpoint = torch.load(opt.modelfile)
-- unpack the model network
vocab = checkpoint.vocab
-- recostruct ivocab
local ivocab = {}
for k,v in pairs(vocab) do ivocab[v] = k end

-- load saved model
model = checkpoint.model
-- reset previous state
model:evaluate() -- switching off BPTT

-- get the next word with max prob given the output probs
function get_max_word(output)
  local p, idx = torch.max(torch.exp(output),1)
  return {ivocab[idx[1]],p}
end

-- sample the next word from multinomial distribution
function get_mult_word(output)
  output:div(opt.temperature) -- scale by temperature
  local probs = torch.exp(output):squeeze()
  probs:div(torch.sum(probs)) -- renormalize so probs sum to one
  local prev_word = torch.multinomial(probs:float(),1)--:resize(1):float()
  nword = {ivocab[torch.totable(prev_word)[1]],}
  if opt.skipunk and nword[1] == "<unk>" then
    nword = get_mult_word(output)
  end
  return nword
end

-- pretty print generated text
function pprint(text_tbl)
  local newsent = true
  for _,w in pairs(text_tbl) do
    if w == "<eos>" or w == "<punct>" then
      -- skip
    elseif newsent and w == '.' then
      newsent = false
    elseif w:gmatch("\n")() then
      io.write(w)
      newsent = true
    else
      io.write(w..' ')
    end
  end
  io.write("\n")
end


-- seed the network with some text before prediction
local generated = {}
for word in seedtext:gmatch("%S+") do
  -- iff in vocab
  if vocab[word] then
    local word_tensor = adddim(torch.IntTensor{vocab[word]})
    model:forward(word_tensor)
    table.insert(generated,word)
  end
end

-- sample new words given previous history
for i=1,opt.length do
  local predicted
  local prev_vec = adddim(torch.IntTensor{vocab[generated[#generated]] or vocab['<eos>']})
  if opt.temperature == -1 then
    predicted = get_max_word(model:forward(prev_vec)[1]:squeeze(1))
  else
    predicted = get_mult_word(model:forward(prev_vec)[1]:squeeze(1))
  end

  if predicted[1] == '<eos>' then
    table.insert(generated,".\n")
  end
  table.insert(generated,predicted[1])
end

pprint(generated)
