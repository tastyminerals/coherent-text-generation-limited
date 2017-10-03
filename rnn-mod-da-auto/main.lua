#!/usr/bin/env th
--[[
This version trains two networks: first is general domain adapted version of topic-events based rnn.
The general network generates probability distribtion over train corpus vocab given topic and event vectors.
Second network is trained to generate probability distribution over event vector vocab.
It uses event vector, topic vector and input word vector for prediction.
<Since we are using lookup layer for inputs in the second network, we'd as well should include word2vec vectors.>

We are using simplified event vector representation in our training corpus, e.g. {174:10, 1:2, 23:3} -> 5.
Simplified encoding is saved in train_num_par.txt and valid_num_par.txt.
It creates a small event vector vocab (18).

Original representation which is based on {174:10, 1:2, 23:3} -> 1, {174:5, 1:2} -> 2, {174:20, 3:5, 7:1} -> 3.
In other words we treat an event vector as a normal token is saved in train_tokvecs.txt and valid_tokvecs.txt.
The original approach creates a huge vocab since the number of unique vectors is high.

]]
package.path = package.path .. ";../?.lua"
require "paths"
require "rnn"
require "math"
require "optim"
require "optim"
local pastalog = require "pastalog"
local network = require "model.network"
local network2 = require "model.network2"
local loader = require "dataloader"
local utils = require "utils.utils"


------------------------ command line arguments ------------------------
cmd = torch.CmdLine()
cmd:text("Model config")
cmd:text()
cmd:text("Options:")

-- data options
cmd:option("--path","data/inscript_red","specify trainset relative path")
cmd:option("--batchsize",32,"number of examples per batch")
cmd:option("--trainsize",-1,"number of training examples seen between epochs")
cmd:option("--validsize",-1,"number of validation examples used for early-stopping and cross-validation")
cmd:option("--savepath",os.getenv("PWD"),"dir path for saving experiment results and model parameters")
cmd:option("--id",'',"experiment output name, defaults to unique id")

-- model options
cmd:option("--rnn",false,"use vanilla RNN instead of GRU")
cmd:option("--seqlen",20,"backpropagation length (rho), how many steps to look back in history")
cmd:option("--hiddensize","{600}","number of hidden units in recurrent layer")
--cmd:option("--inputsize",-1,"size of the lookup table embeddings, -1 defaults to hidden[1]")
cmd:option("--dropout",0,"dropout probability after each model layer, 0 disables dropout")

-- training options
cmd:option("--initlr",0.05,"learning rate at t-time = 0")
cmd:option("--minlr",0.00001,"minimum learning rate")
cmd:option("--saturate",100,"epoch at which linear decay of lr reaches minlr") -- set 400 if not adam
cmd:option("--schedule",'',"learning rate schedule, e.g. '{[5]=0.004,[6]=0.001}'")
cmd:option("--momentum",0.9,"prevents the network from converging to local minimum")
cmd:option('--adam', false, 'use ADAM instead of SGD as optimizer')
cmd:option('--adamconfig', '{0, 0.999}', 'ADAM hyperparameters beta1 and beta2')
cmd:option("--maxnormout",-1,"prevents overfitting, max l2-norm of each layer's output neuron weights")
cmd:option("--cutoff",-1,"max 12-norm of concatenation of all gradParam tensors")
cmd:option("--maxepoch",1000,"max number of epochs to run")
cmd:option("--earlystop",20,"max number of epochs to wait until a better local minima for early-stopping is found")
cmd:option("--uniform",0.1,"init the params using uniform distribution between -uniform and uniform. -1 means default initializaiton")
-- batch normalization with lstm only?

-- gpu options
cmd:option("--cuda",false,"use CUDA")
cmd:option("--device",1,"set device (GPU) number to use")

-- stdout options
cmd:option("--progress",false,"display progress bar")
cmd:option("--silent",false,"suppress stdout")
cmd:text()

-- init accumulated args
local opt = cmd:parse(arg or {})


------------------------ process command line arguments for network 1 ----------------------------
print("\n------------------------[[ Network 1 ! ]]------------------------\n")
opt.hiddensize = loadstring("return "..opt.hiddensize)()
opt.inputsize = 300 -- we shall use word2vec 300 dim embeddings size
opt.schedule = loadstring("return "..opt.schedule)()
opt.adamconfig = loadstring(" return "..opt.adamconfig)()
if opt.silent then table.print(opt) end
opt.id = opt.id == '' and "inscript_"..loader.uniqueid() or opt.id
-- reduce lr if batchsize is 32
if opt.batchsize <= 32 and opt.batchsize > 16 then
  opt.initlr = 0.01
elseif opt.batchsize <=16 and opt.batchsize > 8 then
  opt.initlr = 0.03
end

if opt.cuda then
  require "cunn"
  cutorch.setDevice(opt.device)
end

-- print network hyperparams & params
if not opt.silent then
  for i,v in pairs(opt) do
    if type(v) == 'table' then
      io.write(string.format("%s: ",i))
      print(v)
    else
      print(string.format("%s: %s",i,v))
    end
  end
end
print("")


utils.register(opt, 'opt') -- init opt before using utils.lua


---------------------------- prepare and load dataset for network 1 ----------------------------
-- load training, validation data

local x,x_valid,_ = loader.load_data(opt.path,{opt.batchsize,1,1},'<eos>','[trnvldai]+.txt')
-- get max sent length among all corpus files <required for vectors extraction>
local maxsentlen = math.max(unpack({x.maxsentlen,x_valid.maxsentlen}))

-- (domain adaptation) load pretrained word2vec embeddings for training data
local w2v_train = loader.load_w2v('../word_embeddings/word2vec_inscript.t7',x)
-- (domain adaptation) load pretrained word2vec embeddings for validation data
local w2v_valid = loader.load_w2v('../word_embeddings/word2vec_inscript.t7',x_valid)

-- load training topic & event vectors
local dims1,dims2 = {x.total_tokens,maxsentlen,1},{x_valid.total_tokens,maxsentlen,1}
local topics_train,events_train = loader.load_csv_vectors(opt.path,'train_vecs.csv',opt.batchsize,dims1)
local topics_valid,events_valid = loader.load_csv_vectors(opt.path,'valid_vecs.csv',1,dims2)

assert((x.itersize==topics_train.itersize) and (topics_train.itersize==events_train.itersize),
  'ERROR! inconsistent length of the training data')

-- <used by GRU_mod_nodrop.lua>
opt.egateSize = events_train.data:size(3)
opt.tgateSize = topics_train.data:size(3)

if not opt.silent then
  print("Vocab size: "..#x.ivocab)
  print(string.format("Train data is split into %d sequences of total %d length",opt.batchsize,x:size()))
end


------------------------ create network 1 ------------------------
opt.vecsize = events_train.data:size(3) -- set event vector size

local model = network.build(x,opt)

if not opt.silent then
  print("---------- Language Model 1 ----------")
  print(model)
  print('')
end

-- init params
if opt.uniform > 0 then
  for k,param in ipairs(model:parameters()) do
    param:uniform(-opt.uniform,opt.uniform)
  end
end


------------------------ define network 1 lossfunc function ------------------------
-- SequencerCriterion is a decorator used with Sequencer, applies Criterion to each element of input and target.
local lossfunc = nn.SequencerCriterion(nn.ClassNLLCriterion())
-- build target module, target is also (seqlen x batchsize)
local targetmodule = nn.SplitTable(1)


------------------------ cuda setup ------------------------
if opt.cuda then
  model:cuda()
  lossfunc:cuda()
  targetmodule:cuda()
end


------------------------ configure network 1 experiment log ------------------------
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = opt.path
xplog.vocab = x.vocab
-- will only serialize params
--xplog.model = nn.Serial(model) -- dpnn class that controls model serialization
--xplog.model:mediumSerial() -- set "medium" serialization for lm (recommended)
xplog.criterion = lossfunc
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 0
paths.mkdir(opt.savepath) -- create logging dir


------------------------ train network 1 ------------------------
-- params and grad_params are used later by optim for adam
local params, grad_params = model:getParameters()

local adamconfig = {
  beta1 = opt.adamconfig[1],
  beta2 = opt.adamconfig[2],
}

local ntrial = 0
local epoch = 1
opt.lr = opt.initlr
opt.trainsize = opt.trainsize == -1 and x:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and x_valid:size() or opt.validsize

local begin_time = os.date()
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print('')
  print("Epoch #"..epoch)
  local timer = torch.Timer()

  sgdconfig = {
    learningRate = opt.lr,
    momentum = opt.momentum
    }

  model:training() -- switch the model into training mode

  -- init iterators
  local xit = x:subiter(opt.seqlen,opt.trainsize)
  local w2vtit = w2v_train:subiter(opt.seqlen,opt.trainsize)
  local evit = events_train:subiter(opt.seqlen,opt.trainsize)
  local topit = topics_train:subiter(opt.seqlen,opt.trainsize)

  local err_sum = 0
  for xbatch,w2vbatch,evbatch,topbatch in utils.zip(xit,w2vtit,evit,topit) do
    local i,inputs,targets = unpack(xbatch) -- seqlen x batchsize
    local _,w2vinputs,_ = unpack(w2vbatch) -- word2vec embeddings
    local _,evin,_ = unpack(evbatch) --  seqlen x batchsize x maxsentlen x 1
    local _,topin,_ = unpack(topbatch) --  seqlen x batchsize x 10 x 1

    -- forward pass
    local outputs
    targets = targetmodule:forward(targets)

    local function feval(val)
      if val ~= params then
        params:copy(val)
      end
      grad_params:zero()

      -- forward pass
      outputs = model:forward({inputs,utils.cudify(w2vinputs,true),utils.cudify(evin),utils.cudify(topin)})
      local err = lossfunc:forward(outputs,targets)
      err_sum = err_sum + err

      -- backward pass
      local grad_outputs = lossfunc:backward(outputs, targets)
      model:zeroGradParameters()
      model:backward({inputs,utils.cudify(w2vinputs,true),utils.cudify(evin),utils.cudify(topin)},grad_outputs)

      -- gradient clipping
      if opt.cutoff > 0 then
        local norm = model:gradParamClip(opt.cutoff) -- affects gradParams
        opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end
      return err, grad_params
    end

    -- weight and lr updates happen here
    if opt.adam then
      local _, loss = optim.adam(feval, params, adamconfig)
    else
      local _, loss = optim.sgd(feval, params, sgdconfig)
    end

    -- display progress bar here
    if opt.progress then
      xlua.progress(math.min(i + opt.seqlen,opt.trainsize),opt.trainsize)
    end
    if i%1000 == 0 then collectgarbage() end
  end

  -- learning rate decay
  if opt.schedule then
    opt.lr = opt.schedule[epoch] or opt.lr
  else
    opt.lr = opt.lr + (opt.minlr - opt.initlr)/opt.saturate
  end

  opt.lr = math.max(opt.minlr,opt.lr)


  ------------------------ report network 1 progress ------------------------
  if not opt.silent then
    print(string.format("Network 1 learning rate: %f, cutoff: %d, seqlen: %d",opt.lr,opt.cutoff,opt.seqlen))
    if opt.mean_norm then
      print("Network 1 mean grad_param norm",opt.mean_norm)
    end
  end

  if cutorch then cutorch.synchronize() end

  local speed = timer:time().real/opt.trainsize
  print(string.format("Network 1 elapsed time: %f",timer:time().real))
  print(string.format("Network 1 speed: %f sec/batch",speed))

  local ppl = torch.exp(err_sum/opt.trainsize)
  print("Network 1 PPL before validation: "..ppl)
  xplog.trainppl[epoch] = ppl

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod-da-auto', 'PPL (train)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ network 1 validation ------------------------
  model:evaluate() -- switch the model into eval mode

  -- init validation iterators
  local xvit = x_valid:subiter(opt.seqlen,opt.validsize)
  local w2vvit = w2v_valid:subiter(opt.seqlen,opt.validsize)
  local evvit = events_valid:subiter(opt.seqlen,opt.validsize)
  local topvit = topics_valid:subiter(opt.seqlen,opt.validsize)

  local err_sum = 0
  for xvbatch,w2vvbatch,evvbatch,topvbatch in utils.zip(xvit,w2vvit,evvit,topvit) do
    local i,inputs,targets = unpack(xvbatch)
    local _,w2vin,_ = unpack(w2vvbatch)
    local _,vevin,_ = unpack(evvbatch)
    local _,topvin,_ = unpack(topvbatch)

    targets = targetmodule:forward(targets)
    local outputs = model:forward({inputs,utils.cudify(w2vin,true),utils.cudify(vevin),utils.cudify(topvin)})
    local err = lossfunc:forward(outputs,targets)
    err_sum = err_sum + err
  end

  -- Perplexity (PPL) = exp(sum(NLL)/#2)
  local ppl = torch.exp(err_sum/opt.validsize)
  print("Network 1 PPL : "..ppl)
  xplog.valppl[epoch] = ppl
  ntrial = ntrial + 1

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod-da-auto', 'PPL (val)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ check network 1 early-stopping ------------------------
  if ppl < xplog.minvalppl then
    -- save best version of model
    xplog.minvalppl = ppl
    xplog.epoch = epoch
    filename = paths.concat(opt.savepath,opt.id..".t7")
    print('')
    print("Network 1: found new minima. Saving to "..filename)
    print("")
    torch.save(filename,xplog)
    ntrial = 0
  elseif ntrial >= opt.earlystop then
    print('')
    print("Network 1: no new minima found after "..ntrial.." epochs.")
    print("stopping experiment.")
    break
  end

  collectgarbage() -- free some resources
  epoch = epoch + 1

end

print("\n------------------------[[ FINISHED Network 1 training ! ]]------------------------\n")
print('Began training network 1 at: '.. begin_time)
print('Ended training network 1 at: '..os.date())
print(string.format("Last best epoch: %d, PPL: %f",xplog.epoch,xplog.minvalppl))
print("Model saved: ..".. paths.concat(opt.savepath,opt.id..".t7"))
-- print command used to run main.lua file
utils.run_command(arg, '\nCalled with: th main.lua ')
print("\n")


------------------------------------------------------------------------------------------------
print("\n------------------------[[ Network 2 ! ]]------------------------\n")
local opt2 = utils.clone(opt)
-- set params for parallel network
opt2.hiddensize = {50} -- smaller hidden layer since our vocab is small
opt2.inputsize = opt.inputsize -- x input vectors dim, 300
opt2.schedule = opt.schedule
--opt2.batchsize = 20
--opt2.seqlen = 20
opt2.id = "inscript_net2_"..loader.uniqueid()


-- print network hyperparams & params
if not opt2.silent then
  for i,v in pairs(opt2) do
    if type(v) == 'table' then
      io.write(string.format("%s: ",i))
      print(v)
    else
      print(string.format("%s: %s",i,v))
    end
  end
end
print("")


---------------------------- prepare and load dataset for network 2 ----------------------------
local x2,x2_valid,_ = loader.load_data(opt.path,{opt2.batchsize,1,1},' ','[trnvldai]_decay+.txt')

-- (domain adaptation) load pretrained word2vec embeddings for training data
local w2v_train = loader.load_w2v('../word_embeddings/word2vec_inscript.t7',x)
-- (domain adaptation) load pretrained word2vec embeddings for validation data
local w2v_valid = loader.load_w2v('../word_embeddings/word2vec_inscript.t7',x_valid)

-- load training topic & event vectors
local dims1,dims2 = {x2.total_tokens,maxsentlen,1},{x2_valid.total_tokens,maxsentlen,1}
local topics_train,_ = loader.load_csv_vectors(opt.path,'train_vecs.csv',opt2.batchsize,dims1)
local topics_valid,_ = loader.load_csv_vectors(opt.path,'valid_vecs.csv',1,dims2)

-- used by GRU_mod.lua
opt2.egateSize = opt2.inputsize
opt2.tgateSize = opt.tgateSize

if not opt.silent then
  print("Vocab size: "..#x2.ivocab)
  print(string.format("Train data is split into %d sequences of total %d length",opt2.batchsize,x2:size()))
end


------------------------ create model network 2 ------------------------
local model2 = network2.build(x2,opt2)

if not opt2.silent then
  print("---------- Language Model 2 ----------")
  print(model2)
  print('')
end

-- init params
if opt2.uniform > 0 then
  for k,param2 in ipairs(model2:parameters()) do
    param2:uniform(-opt2.uniform,opt2.uniform)
  end
end


------------------------ define network 2 lossfunc function ------------------------
-- SequencerCriterion is a decorator used with Sequencer, applies Criterion to each element of input and target.
local lossfunc2 = nn.SequencerCriterion(nn.ClassNLLCriterion())
-- build target module, target is also (seqlen x batchsize)
local targetmodule2 = nn.SplitTable(1)


------------------------ cuda setup ------------------------
if opt2.cuda then
  model2:cuda()
  lossfunc2:cuda()
  targetmodule2:cuda()
end


------------------------ configure network 2 experiment log ------------------------
local xplog2 = {}
xplog2.opt = opt2
xplog2.dataset = opt.path
xplog2.vocab = x2.vocab
xplog2.model = nn.Serial(model2)
xplog2.model:mediumSerial()
xplog2.criterion = lossfunc2
xplog2.targetmodule = targetmodule2
xplog2.trainppl = {}
xplog2.valppl = {}
xplog2.minvalppl = 99999999
xplog2.epoch = 0
paths.mkdir(opt.savepath) -- create logging dir


------------------------ train network 2 ------------------------
-- params and grad_params are used later by optim for adam
local params, grad_params = model2:getParameters()

local adamconfig = {
  beta1 = opt.adamconfig[1],
  beta2 = opt.adamconfig[2],
}

opt2.lr = opt2.initlr
opt2.trainsize = opt.trainsize == -1 and x2:size() or opt.trainsize
opt2.validsize = opt.validsize == -1 and x2_valid:size() or opt.validsize
local ntrial = 0
local epoch = 1

local begin_time = os.date()
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print('')
  print("Epoch #"..epoch)
  local timer = torch.Timer()

  sgdconfig = {
    learningRate = opt.lr,
    momentum = opt.momentum
    }

  model2:training() -- switch the model into training mode

  -- init iterators
  local x2it = x2:subiter(opt2.seqlen,opt2.trainsize)
  local w2vtit = w2v_train:subiter(opt2.seqlen,opt2.trainsize)
  local topit = topics_train:subiter(opt2.seqlen,opt2.trainsize)

  local err_sum = 0
  for x2batch,w2vbatch,topbatch in utils.zip(x2it,w2vtit,topit) do
    local i,inputs,targets = unpack(x2batch) -- seqlen x batchsize
    local _,w2vin,_ = unpack(w2vbatch) -- word2vec embeddings
    local _,topin,_ = unpack(topbatch) --  seqlen x batchsize x 10 x 1

    -- forward pass
    local outputs
    targets = targetmodule2:forward(targets)

    local function feval(val)
      if val ~= params then
        params:copy(val)
      end
      grad_params:zero()

      -- forward pass
      outputs = model2:forward({inputs,utils.cudify(w2vin,true),utils.cudify(topin)})
      local err = lossfunc2:forward(outputs,targets)
      err_sum = err_sum + err

      -- backward pass
      local grad_outputs = lossfunc2:backward(outputs, targets)
      model2:zeroGradParameters()
      model2:backward({inputs,utils.cudify(w2vin,true),utils.cudify(topin)},grad_outputs)

      -- gradient clipping
      if opt2.cutoff > 0 then
        local norm = model2:gradParamClip(opt2.cutoff) -- affects gradParams
        opt2.meanNorm = opt2.meanNorm and (opt2.meanNorm*0.9 + norm*0.1) or norm
      end
      return err, grad_params
    end

    -- weight and lr updates happen here
    if opt2.adam then
      local _, loss = optim.adam(feval, params, adamconfig)
    else
      local _, loss = optim.sgd(feval, params, sgdconfig)
    end

    -- display progress bar here
    if opt.progress then
      xlua.progress(math.min(i + opt2.seqlen,opt2.trainsize),opt2.trainsize)
    end
    if i%1000 == 0 then collectgarbage() end
  end

  -- learning rate decay
  if opt2.schedule then
    opt2.lr = opt2.schedule[epoch] or opt2.lr
  else
    opt2.lr = opt2.lr + (opt2.minlr - opt2.initlr)/opt2.saturate
  end

  opt2.lr = math.max(opt2.minlr,opt2.lr)


  ------------------------ report network 2 progress ------------------------
  if not opt2.silent then
    print(string.format("Network 2 learning rate: %f, cutoff: %d, seqlen: %d",opt2.lr,opt2.cutoff,opt2.seqlen))
    if opt2.mean_norm then
      print("Network 2 mean grad_param norm",opt2.mean_norm)
    end
  end

  if cutorch then cutorch.synchronize() end

  local speed = timer:time().real/opt2.trainsize
  print(string.format("Network 2 elapsed time: %f",timer:time().real))
  print(string.format("Network 2 speed: %f sec/batch",speed))

  local ppl = torch.exp(err_sum/opt2.trainsize)
  print("Network 2 PPL before validation: "..ppl)
  xplog2.trainppl[epoch] = ppl

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod-da-auto epred', 'PPL (train)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ network 2 validation ------------------------
  model2:evaluate() -- switch the model into eval mode

  -- init validation iterators
  local x2vit = x2_valid:subiter(opt2.seqlen,opt2.trainsize)
  local w2vvtit = w2v_valid:subiter(opt2.seqlen,opt2.trainsize)
  local topvit = topics_valid:subiter(opt2.seqlen,opt2.trainsize)

  -- set validation error sum
  local err_sum = 0
  for x2vbatch,w2vbatch,topvbatch in utils.zip(x2vit,w2vvtit,topvit) do
    local i,inputs,targets = unpack(x2vbatch) -- seqlen x batchsize
    local _,w2vvin,_ = unpack(w2vbatch) -- word2vec embeddings
    local _,topvin,_ = unpack(topvbatch) --  seqlen x batchsize x 10 x 1

    -- forward pass

    if inputs:size(1) ~= opt2.seqlen then -- in case of bad train/valid split
      break
    end
    targets = targetmodule2:forward(targets)
    local outputs = model2:forward({inputs,utils.cudify(w2vvin,true),utils.cudify(topvin)})
    local err = lossfunc2:forward(outputs,targets)
    err_sum = err_sum + err
  end

  -- Perplexity = exp(sum(NLL)/#2)
  local ppl = torch.exp(err_sum/opt2.validsize)
  print("Training PPL : "..ppl)
  xplog2.valppl[epoch] = ppl
  ntrial = ntrial + 1


  ------------------------ report network 2 progress ------------------------
  -- API is pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod-da-auto epred', 'PPL (val)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ check network 2 early-stopping ------------------------
  if ppl < xplog2.minvalppl then
    -- save best version of model
    xplog2.minvalppl = ppl
    xplog2.epoch = epoch
    filename2 = paths.concat(opt2.savepath,opt2.id..".t7")
    print("Found new minima. Saving to "..filename2)
    print("")
    torch.save(filename2,xplog2)
    ntrial = 0
  elseif ntrial >= opt2.earlystop then
    print("No new minima found after "..ntrial.." epochs.")
    print("stopping experiment.")
    break
  end

  -- if PPL is ~1, break
  if ppl <= 1.1 then
    print('PPL reached '..ppl)
    print('stopping experiment.')
    break
  end

  collectgarbage() -- free some resources
  epoch = epoch + 1

end

print("\n------------------------[[ FINISHED Network 2 training ! ]]------------------------\n")
print('Began training network 2 at: '.. begin_time)
print('Ended training network 2 at: '..os.date())
print(string.format("Last best epoch: %d, PPL: %f",xplog2.epoch,xplog2.minvalppl))
print("Model saved: ..".. paths.concat(opt2.savepath,opt2.id..".t7"))
-- print command used to run main.lua file
utils.run_command(arg, '\nCalled with: th main.lua ')

print('\nNetwork 1 best PPL: '..xplog.minvalppl)
print('Saved network 1: '..filename)
print('')
print('\nNetwork 2 best PPL: '..xplog2.minvalppl)
print('Saved network 2: '..filename2)



