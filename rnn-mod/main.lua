#!/usr/bin/env th
--[[
A modification of the classic GRU lm that uses event and topic information provided by
InScript annotations in addition to input vectors.
]]

require "paths"
require "rnn"
require "optim"
local pastalog = require "pastalog"
local network = require "model.network"
local loader = require "dataloader"


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
cmd:option("--seqlen",20,"backpropagation length, how many steps to look back in history")
cmd:option("--hiddensize","{400}","number of hidden units per recurrent layer")
cmd:option("--inputsize",-1,"size of the lookup table embeddings, -1 defaults to hidden[1]")
cmd:option("--dropout",0.5,"dropout probability after each model layer, 0 disables dropout")

-- training options
cmd:option("--initlr",0.05,"learning rate at t-time = 0")
cmd:option("--minlr",0.00001,"minimum learning rate")
cmd:option("--saturate",100,"epoch at which linear decay of lr reaches minlr")
cmd:option("--schedule",'',"learning rate schedule, e.g. '{[5]=0.004,[6]=0.001}'")
cmd:option("--momentum",0.9,"prevents the network from converging to local minimum")
cmd:option('--adam', false, 'use ADAM instead of SGD as optimizer')
cmd:option('--adamconfig', '{0, 0.999}', 'ADAM hyperparameters beta1 and beta2')
cmd:option("--maxnormout",-1,"prevents overfitting, max 12-norm of each layer's output neuron weights")
cmd:option("--cutoff",10,"max 12-norm of concatenation of all gradParam tensors")
cmd:option("--maxepoch",200,"max number of epochs to run")
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

local opt = cmd:parse(arg or {})


-------------------- helper functions --------------------
-- DEBUG ONLY
function px(...)
  print(...)
  os.exit()
end

-- print command line arguments used to start the script
function run_command()
  local sorted = {}
  for k,v in ipairs(arg) do
    table.insert(sorted,v)
  end
  for k,v in ipairs(sorted) do
    if k > 0 then
      io.write(string.format("%s ",v))
    end
  end
  io.write("\n")
end

function zip(...)
  local arrays, ans = {...}, {}
  local index = 0
  return function()
    index = index + 1
    for i,t in ipairs(arrays) do
      if type(t) == 'function' then
        local nsample,inputs,targets = t()
        if nsample == nil then return end
        ans[i] = {nsample,inputs,targets}
      else
        ans[i] = t[index]
        if ans[i] == nil then return end
      end
    end
    return unpack(ans)
  end
end

-- remove 4th dim and check if cuda was enabled
function cudify(vecs)
  if opt.cuda then
    return vecs:squeeze(4):cuda()
  else
    return vecs:squeeze(4):double()
  end
end


-------------------- process command line arguments --------------------
opt.hiddensize = loadstring("return "..opt.hiddensize)()
opt.inputsize = opt.inputsize == -1 and opt.hiddensize[1] or opt.inputsize
opt.schedule = loadstring("return "..opt.schedule)()
opt.adamconfig = loadstring(" return "..opt.adamconfig)()
if opt.silent then table.print(opt) end
opt.id = opt.id == '' and "inscript_"..loader.uniqueid() or opt.id
if opt.cuda then
  require "cunn"
  cutorch.setDevice(opt.device)
end

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


---------------------------- prepare and load dataset for network ----------------------------
-- load training, validation data
local trainfiles = {'train.txt', 'valid.txt'}
local x,x_valid,_ = loader.load_data(opt.path,{opt.batchsize,1,1},'<eos>',trainfiles)
-- get max sent length among all corpus files <required for vectors extraction>
local maxsentlen = math.max(unpack({x.maxsentlen,x_valid.maxsentlen}))

-- load training topic & event vectors
local dims1,dims2 = {x.total_x,maxsentlen,1},{x_valid.total_x,maxsentlen,1}
local topics,events = loader.load_csv_vectors(opt.path,'train_vecs.csv',opt.batchsize,dims1)
local topics_valid,events_valid = loader.load_csv_vectors(opt.path,'valid_vecs.csv',1,dims2)

assert((x.itersize==topics.itersize), 'ERROR! inconsistent length of the training data')
assert((topics.itersize==events.itersize), 'ERROR! inconsistent length of the training data')
assert((topics_valid.itersize==events_valid.itersize), 'ERROR! inconsistent length of the training data')

if not opt.silent then
  print("Vocab size: "..#x.ivocab)
  print(string.format("Train data is split into %d sequences of total %d length",opt.batchsize,x:size()))
end

-- <used by GRU_mod.lua>
opt.egateSize = events.data:size(3)
opt.tgateSize = topics.data:size(3)


------------------------ create network ------------------------
opt.vecsize = events.data:size(3) -- set event vector size
local model = network.build(x,opt)

if not opt.silent then
  print("Language Model:")
  print(model)
end

-- init params
if opt.uniform > 0 then
  for k,param in ipairs(model:parameters()) do
    param:uniform(-opt.uniform,opt.uniform)
  end
end


-------------------- define lossfunc function --------------------
-- SequencerCriterion is a decorator used with Sequencer, applies Criterion to each element of input and target.
local lossfunc = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- build target module, target is also (seqlen x batchsize)
local targetmodule = nn.SplitTable(1)


-------------------- cuda setup --------------------
if opt.cuda then
  model:cuda()
  lossfunc:cuda()
  targetmodule:cuda()
end


-------------------- configure experiment log --------------------
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = "InScript corpus"
xplog.vocab = x.vocab
-- will only serialize params
xplog.model = nn.Serial(model) -- dpnn class that controls model serialization
xplog.model:mediumSerial() -- set "medium" serialization for lm (recommended)
xplog.criterion = lossfunc
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 0

paths.mkdir(opt.savepath) -- create logging dir


-------------------- train the model --------------------
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
  print()
  print("Epoch #"..epoch)
  local timer = torch.Timer()

  sgdconfig = {
    learningRate = opt.lr,
    momentum = opt.momentum
    }

  model:training() -- switch the model into training mode

  -- init iterators
  local xit = x:subiter(opt.seqlen,opt.trainsize)
  local topicsit = topics:subiter(opt.seqlen,opt.trainsize)
  local eventsit = events:subiter(opt.seqlen,opt.trainsize)

  local err_sum = 0
  for tokbatch,topbatch,evbatch in zip(xit,topicsit,eventsit) do
    local i,inputs,targets = unpack(tokbatch) -- 5x32
    local _,topin,_ = unpack(topbatch) -- 5x32x10x1
    local _,evin,_ = unpack(evbatch) -- 5x32x53x1

    targets = targetmodule:forward(targets)


    local function feval(val)
      if val ~= params then
        params:copy(val)
      end
      grad_params:zero()

      -- forward pass
      local outputs = model:forward({inputs,cudify(evin),cudify(topin)})
      local err = lossfunc:forward(outputs, targets)
      err_sum = err_sum + err

      -- backward pass
      local grad_outputs = lossfunc:backward(outputs, targets)
      model:zeroGradParameters()
      model:backward({inputs,cudify(evin),cudify(topin)},grad_outputs)

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


  ------------------------ report progress ------------------------
  if not opt.silent then
    print(string.format("learning rate: %f, cutoff: %d, seqlen: %d",opt.lr,opt.cutoff,opt.seqlen))
    if opt.mean_norm then
      print("mean grad_param norm",opt.mean_norm)
    end
  end

  if cutorch then cutorch.synchronize() end

  local speed = timer:time().real/opt.trainsize
  print(string.format("Elapsed time: %f",timer:time().real))
  print(string.format("Speed: %f sec/batch",speed))

  local ppl = torch.exp(err_sum/opt.trainsize)
  print("Training PPL before cross-validation: "..ppl)
  xplog.trainppl[epoch] = ppl

  -- occasional weight blow ups screw pastalog training curves
  if xplog.trainppl[epoch-1] and ppl > xplog.trainppl[epoch-1] * 10 then
    print('>>> BOOM! your weights blew through the celining. Using the last ppl score.')
    ppl = xplog.trainppl[epoch-1] -- use last ppl score instead
  end

  -- API is pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod', 'PPL (training)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ validation ------------------------
  model:evaluate() -- switch the model into eval mode

  local vxit = x_valid:subiter(opt.seqlen,opt.validsize)
  local vtopicsit = topics_valid:subiter(opt.seqlen,opt.validsize)
  local veventsit = events_valid:subiter(opt.seqlen,opt.validsize)

  local err_sum = 0

  local a = 0
  for vtoks,vtopics,vevents in zip(vxit,vtopicsit,veventsit) do
    local i,inputs,targets = unpack(vtoks)
    local _,vtopin,_ = unpack(vtopics)
    local _,vevin,_ = unpack(vevents)

    targets = targetmodule:forward(targets)
    local outputs = model:forward({inputs,cudify(vevin),cudify(vtopin)})
    local err = lossfunc:forward(outputs,targets)

    err_sum = err_sum + err
  end

  -- Perplexity = exp(sum(NLL)/#2)
  local ppl = torch.exp(err_sum/opt.validsize)
  print("Training PPL : "..ppl)
  xplog.valppl[epoch] = ppl
  ntrial = ntrial + 1

  -- occasional weight blow ups screw pastalog training curves
  if xplog.valppl[epoch-1] and ppl > xplog.valppl[epoch-1] * 10 then
    print('>>> BOOM! your weights blew through the celining. Using the last ppl score.')
    ppl = xplog.valppl[epoch-1] -- use last ppl score instead
  end

  -- API is pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod', 'PPL (validation)', ppl, epoch, 'http://localhost:8120/data')

  -- early-stopping
  if ppl < xplog.minvalppl then
    -- save best version of model
    xplog.minvalppl = ppl
    xplog.epoch = epoch
    local filename = paths.concat(opt.savepath,opt.id..".t7")
    print("Found new minima. Saving to "..filename)
    print("")
    torch.save(filename,xplog)
    ntrial = 0
  elseif ntrial >= opt.earlystop then
    print("No new minima found after "..ntrial.." epochs.")
    print("stopping experiment.")
    break
  end

  collectgarbage() -- free some resources
  epoch = epoch + 1

end

print('Began training at: '.. begin_time)
print('Ended training at: '..os.date())
print(string.format("Last best epoch: %d, PPL: %f",xplog.epoch,xplog.minvalppl))
print("\nEvaluate model using : ")
print("\tth ../samples/evaluate-rnnlm.lua --xplogpath "..paths.concat(opt.savepath,opt.id..".t7")..(opt.cuda and " --cuda" or ""))
