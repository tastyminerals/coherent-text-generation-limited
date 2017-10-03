#!/usr/bin/env th
--[[
rnn-mod-da requires event and topic vectors to be created artificially.
During generation the event vector will be decayed according to the dictionary of event labeled tokens.
]]
package.path = package.path .. ";../?.lua"
require "rnn"
require "model.GRU_mod"

local file = require "pl.file"
local path = require "pl.path"
local JSON = require "JSON" -- luarocks install lua-json
local stringx = require "pl.stringx"
local dlr = require "dataloader"
local utils = require "utils.utils"
local adddim = nn.utils.addSingletonDimension

assert(path.isfile('labeled_tokens.json'), 'Required "labeled_tokens.json" not found!')

-- command line arguments
cmd = torch.CmdLine()
cmd:text("Sampler config")
cmd:text()
cmd:text("Options:")

-- model argument, required
cmd:argument("-modelfile","specify saved model checkpoint to load")
cmd:argument("-epredictor","specify saved event predictor checkpoint to load")
cmd:argument("-seedfile","specify a file containing seed text, event/topic vectors")

-- text generation options, optional
cmd:option("--seed",123,"a number used to initialize rnd generator")
cmd:option("--seedtext",'',"specify seed text, will be used instead of 'initial' seedfile text")
cmd:option("--topic",'',"specify text topic, will be used instead of 'miniplan' seedfile topic")
cmd:option("--temperature",-1,"set sampling temperature [0-1] (higher, more diversity and errors), if not set uses maxprob sampling")
cmd:option("--skipunk",false,"skip <unk> words during generation")
cmd:option("--cuda",false,"use GPU")
cmd:text()

-- init accumulated args
local opt = cmd:parse(arg or {})
utils.register(opt, 'opt')

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(1)
end


-- decrements event vector
function decay(vec,num) -- torch tensor, number to nullify
  local dvec = vec:clone()
  for i=1,dvec:size(1) do
    if dvec[i] == num then -- find the first match
      dvec[i] = -1 -- mark it
      break
    end
  end
  -- :gt :lt :ge :le :eq
  if torch.sum(dvec:eq(-1)) == 1 then
    dvec = dvec[dvec:ge(0)]:cat(torch.IntTensor{0}) -- append 0
  end
  return dvec
end


-- set rnd seed
torch.manualSeed(opt.seed)
-- load saved checkpoints
local checkpoint = torch.load(opt.modelfile)
local checkpoint2 = torch.load(opt.epredictor)
-- unpack the model network
local vocab = checkpoint.vocab
local evocab = checkpoint2.vocab
utils.register(vocab, 'vocab')
utils.register(evocab, 'evocab')
-- recostruct ivocab
local ivocab, eivocab = {}, {}
for k,v in pairs(vocab) do ivocab[v] = k end
for k,v in pairs(evocab) do eivocab[v] = k end
utils.register(ivocab, 'ivocab')
utils.register(eivocab, 'eivocab')
-- set model
local model = checkpoint.model
local epredictor = checkpoint2.model
-- reset previous state
model:forget()
epredictor:forget()
model:evaluate() -- switching off BPTT
epredictor:evaluate() -- switching off BPTT
utils.register(model, 'model')
utils.register(epredictor, 'epredictor')

-- load seedfile
local seedfile = JSON:decode(file.read('seedfile.json'))
-- load y_t vectors
local w2v = dlr.word2vec('../word_embeddings/word2vec_inscript.t7')

-- construct x_t table
local x_t = {}
if opt.seedtext == '' then
  for i=1,#seedfile['initial'] do
    table.insert(x_t, seedfile['initial'][i])
  end
else
  local seedtext = stringx.split(opt.seedtext) -- injecting opt.seedtext for collect_gen.lua
  for i=1,#seedtext do
    table.insert(x_t, seedtext[i])
  end
end

-- construct e_t, t_t vectors, we shall still be using the seedfile to support the epredictor
local e_t, t_t, curtopic = {}, {}
for i=1,#seedfile['miniplans'] do
  local line_spl = stringx.split(seedfile['miniplans'][i])
  table.insert(e_t, utils.make_evector(line_spl[2], line_spl[3]))
  curtopic = function() if opt.topic ~= '' then return opt.topic else return line_spl[1] end end -- injecting opt.topic for collect_gen.lua
  table.insert(t_t, utils.topic2vec(curtopic(),10))
end

assert((#e_t == #t_t), 'ERROR: Inconsistent vectors!')


---------------------------------------- SEEDING ----------------------------------------
local generated = {}
for i,token in ipairs(x_t) do
  token = string.lower(token)
  if vocab[token] then
    local input_x = adddim(torch.IntTensor{vocab[token]})
    local w2v_vec = adddim(w2v[token] or torch.FloatTensor(1,300):fill(0))
    model:forward({input_x,
                   utils.tocuda(w2v_vec, true),
                   utils.tocuda(adddim(adddim(e_t[1]))), -- we feed the same event vector without decaying it
                   utils.tocuda(adddim(adddim(t_t[1])))}) -- we do not need to change the topic vector

    -- model2:forward({inputs,utils.cudify(w2vin,true),utils.cudify(topin)})
    -- convert e_t into "2-24" representation
    epredictor:forward({adddim(torch.IntTensor{evocab[utils.epred_vector2token(e_t[1])]}), -- evec -> "2-24" -> vocab idx
                        utils.tocuda(w2v_vec,true),
                        utils.tocuda(adddim(adddim(t_t[1])))})

    table.insert(generated,token) -- keep seeded text
  else
    print(string.format('WARNING! "%s" not in the vocab, skipping',token))
  end
end


---------------------------------------- GENERATION ----------------------------------------
local prev_e, prev_t = e_t[2], t_t[2] -- set prev vector vars

for i=1,#seedfile['miniplans'] do -- generate the number of sents specified in seedfile
  local etoks, words = {}, {}
  repeat
    local predicted
    -- get previous token vector
    local prev_x = adddim(torch.IntTensor{vocab[generated[#generated]]})
    -- get word2vec vector
    local w2v_vec = adddim(w2v[generated[#generated]] or torch.FloatTensor(1,300):fill(0))

    -- forward pass
    local probs = model:forward({prev_x, -- x_t
                                 utils.tocuda(w2v_vec, true), -- y_t
                                 utils.tocuda(adddim(adddim(prev_e))), -- e_t
                                 utils.tocuda(adddim(adddim(prev_t)))})[1] -- t_t


    local eidx = evocab[utils.epred_vector2token(prev_e)] or 20 --> evec -> "2-24" --> evocab idx
    local eprobs = epredictor:forward({adddim(torch.IntTensor{eidx}),
                                       utils.tocuda(w2v_vec,true),
                                       utils.tocuda(adddim(adddim(prev_t)))})[1]

    local etok
    -- use multivar or max sampling
    if opt.temperature == -1 then
      predicted = utils.get_max_word(probs)
      etok = utils.get_max_eword(eprobs)[1]
      prev_e = utils.epred_token2vector(etok)
    else
      predicted = utils.get_mult_word(probs)
      etok = utils.get_mult_eword(eprobs)[1]
      prev_e = utils.epred_token2vector(etok)
    end

    -- detect if epredictor is got stuck
    if not etoks[etok] then
      etoks[etok] = 1
    else
      etoks[etok] = etoks[etok] + 1
    end

    -- if it stuck, check whether it is event label or NO EVENT label and decay
    if etoks[etok] > 6 then -- higher value longer sentences
      if stringx.split(etok,'-')[1] ~= '0' then
        prev_e = decay(prev_e, 5)
      else
        prev_e = decay(prev_e, 1)
      end
    end


    -- detect if got stuck
    if not words[predicted[1]] then
      words[predicted[1]] = 1
    else
      words[predicted[1]] = words[predicted[1]] + 1
    end

    -- if it stuck, check whether it is event label or NO EVENT label and decay
    if words[predicted[1]] > 5 then -- higher value longer sentences
      prev_e = decay(prev_e, 5)
      prev_e = decay(prev_e, 1)
      predicted[1] = '<eos>'
      words = {}
    end


    --if torch.sum(prev_e) == 0 then
    if predicted[1] == '<eos>' then
      words = {}
      table.insert(generated,"\n")
      -- update topic and event vectors once the current event vector is 0
      prev_e = e_t[i] -- get next event vector
      prev_t = t_t[i] -- get next topic vector
      --pprint(generated) os.exit()
    end
    table.insert(generated,predicted[1])
  until predicted[1] == '<eos>'
end

utils.pprint(generated)
