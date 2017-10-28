#!/usr/bin/env th
--[[
This is a copy of rnn_mod_da without event vectors.
]]
package.path = package.path .. ";../?.lua"
require "paths"
require "rnn"
require "math"
require "optim"
local pastalog = require "pastalog"
local network = require "model.network"
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
cmd:option("--hiddensize","{400}","number of hidden units in recurrent layer")
--cmd:option("--inputsize",-1,"size of the lookup table embeddings, -1 defaults to hidden[1]")
cmd:option("--dropout",0.5,"dropout probability after each model layer, 0 disables dropout")

-- training options
cmd:option("--initlr",0.05,"learning rate at t-time = 0")
cmd:option("--minlr",0.00001,"minimum learning rate")
cmd:option("--saturate",100,"epoch at which linear decay of lr reaches minlr")
cmd:option("--schedule",'',"learning rate schedule, e.g. '{[5]=0.004,[6]=0.001}'")
cmd:option("--momentum",0.9,"prevents the network from converging to local minimum")
cmd:option('--adam', false, 'use ADAM instead of SGD as optimizer')
cmd:option('--adamconfig', '{0, 0.999}', 'ADAM hyperparameters beta1 and beta2')
cmd:option("--maxnormout",-1,"prevents overfitting, max l2-norm of each layer's output neuron weights")
cmd:option("--cutoff",15,"max 12-norm of concatenation of all gradParam tensors")
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

-- init accumulated args
local opt = cmd:parse(arg or {})


------------------------ process command line arguments for network 1 ----------------------------
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
local x,x_valid,_ = loader.load_data(opt.path,{opt.batchsize,1,1},'<eos>',{'train.txt','valid.txt'})
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
      outputs = model:forward({inputs,utils.cudify(w2vinputs,true),
                               utils.zero(utils.cudify(evin)),
                               utils.cudify(topin)})
      local err = lossfunc:forward(outputs,targets)
      err_sum = err_sum + err

      -- backward pass
      local grad_outputs = lossfunc:backward(outputs, targets)
      model:zeroGradParameters()
      model:backward({inputs,utils.cudify(w2vinputs,true),
                      utils.zero(utils.cudify(evin)),
                      utils.cudify(topin)},grad_outputs)

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

  -- occasional weight blow ups screw pastalog training curves
  if xplog.trainppl[epoch-1] and ppl > xplog.trainppl[epoch-1] * 10 then
    print('>>> BOOM! your weights blew through the celining. Using the last ppl score.')
    ppl = xplog.trainppl[epoch-1] -- use last ppl score instead
    print("Network 1 PPL after BOOM: "..ppl)
  end

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod-topic', 'PPL (train)', ppl, epoch, 'http://localhost:8120/data')


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
    local outputs = model:forward({inputs,utils.cudify(w2vin,true),
                                   utils.zero(utils.cudify(vevin)),
                                   utils.cudify(topvin)})
    local err = lossfunc:forward(outputs,targets)
    err_sum = err_sum + err
  end

  -- Perplexity (PPL) = exp(sum(NLL)/#2)
  local ppl = torch.exp(err_sum/opt.validsize)
  print("Network 1 PPL : "..ppl)
  xplog.valppl[epoch] = ppl
  ntrial = ntrial + 1

  -- occasional weight blow ups screw pastalog training curves
  if xplog.valppl[epoch-1] and ppl > xplog.valppl[epoch-1] * 10 then
    print('>>> BOOM! your weights blew through the celining. Using the last ppl score.')
    ppl = xplog.valppl[epoch-1] -- use last ppl score instead
    print("Network 1 PPL after BOOM: "..ppl)
  end

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-mod-topic', 'PPL (val)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ check network 1 early-stopping ------------------------
  if ppl < xplog.minvalppl then
    -- save best version of model
    xplog.minvalppl = ppl
    xplog.epoch = epoch
    local filename = paths.concat(opt.savepath,opt.id..".t7")
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
