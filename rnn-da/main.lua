--[[
This is GRU lm with fast domain adaptation.
Input vectors are summed with pretrained word2vec embeddings.
]]
require "paths"
require "rnn"
require "optim"
local loader = require "loader.dataloader"
local pastalog = require 'pastalog'


------------------------ command line arguments ------------------------
cmd = torch.CmdLine()
cmd:text("Model config")
cmd:text()
cmd:text("Options:")

-- data options
cmd:option("--path","data/inscript","specify trainset relative path")
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
cmd:option("--earlystop",50,"max number of epochs to wait until a better local minima for early-stopping is found")
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


------------------------ helper functions ------------------------
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
    return vecs:cuda()
  else
    return vecs:double()
  end
end


------------------------ process command line arguments ----------------------------
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
local x,x_valid,_ = loader.load_tokens(opt.path,{opt.batchsize,1,1})

-- load pretrained word2vec embeddings
local w2v_path = "../word_embeddings/word2vec_inscript.t7"
local w2v_train = loader.load_w2v(w2v_path,x)
local w2v_valid = loader.load_w2v(w2v_path,x_valid)
assert(x.data:size(1) == w2v_train.data:size(1))
assert(x_valid.data:size(1) == w2v_valid.data:size(1))

if not opt.silent then
  print("Vocab size: "..#x.ivocab)
  print(string.format("Train data is split into %d sequences of total %d length",opt.batchsize,x:size()))
end


------------------------ create network ------------------------
local model = nn.Sequential()

local p = nn.ParallelTable()
-- build input layer as lookup table
local lookup = nn.LookupTable(#x.ivocab,300) -- word2vec uses 300
lookup.maxNorm = -1 -- prevents weird maxnormout behaviour
p:add(lookup)
p:add(nn.Identity())
model:add(p) -- input layer has dimensions (seqlen x batchsize)

model:add(nn.CAddTable()) -- WVS domain adaptation
-- gru has dropout, add if rnn was selected
if opt.rnn then model:add(nn.Dropout(opt.dropout)) end

model:add(nn.SplitTable(1))

-- build recurrent layers
local stepmodule = nn.Sequential() -- a module which is applied as each time t step
local inputsize = opt.inputsize

-- since we can supply more than one rnn layer, we do a loop
for i,hiddensize in ipairs(opt.hiddensize) do
  local rnn
  if not opt.rnn then
    rnn = nn.GRU(300,hiddensize,nil,opt.dropout/2)
  else
    local rm = nn.Sequential() -- input is {x[t],h[t-1]}
    local pt = nn.ParallelTable()
    pt:add(i==1 and nn.Identity() or nn.Linear(300,hiddensize)) -- input layer
    pt:add(nn.Linear(hiddensize,hiddensize))
    rm:add(pt)
    rm:add(nn.CAddTable()) -- merge outputs of ParallelTable
    rm:add(nn.Sigmoid()) -- add non-linearity
    rnn = nn.Recurrence(rm,hiddensize,1) -- (module, outputsize, nInputDim)
  end
  stepmodule:add(rnn) -- stacking layers
  if opt.dropout > 0 then
    stepmodule:add(nn.Dropout(opt.dropout))
  end
  inputsize = hiddensize
end

-- output layer, since we need prob over complete vocab, our output layer has the same dims
stepmodule:add(nn.Linear(inputsize,#x.ivocab))
stepmodule:add(nn.LogSoftMax())

-- wrapping stepmodule into Sequencer
model:add(nn.Sequencer(stepmodule))

-- remember previous state between batches
model:remember((opt.rnn and "eval") or "both")

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


------------------------ define loss function ------------------------
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


------------------------ configure network experiment log ------------------------
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
local ntrial = 0
paths.mkdir(opt.savepath) -- create logging dir


------------------------ train network ------------------------
-- params and grad_params are used later by optim for adam
local params, grad_params = model:getParameters()

local adamconfig = {
  beta1 = opt.adamconfig[1],
  beta2 = opt.adamconfig[2],
}

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
  local trainit = x:subiter(opt.seqlen,opt.trainsize)
  local w2vtit = w2v_train:subiter(opt.seqlen,opt.trainsize)

  local err_sum = 0
  for trainbatch,w2vbatch in zip(trainit,w2vtit) do
    --for i,inputs,targets in x:subiter(opt.seqlen,opt.trainsize) do
    local i,inputs,targets = unpack(trainbatch)
    local _,w2vinputs,_ = unpack(w2vbatch)

    targets = targetmodule:forward(targets)  -- targets (IntTensor 5x32)

    local function feval(val)
      if val ~= params then
        params:copy(val)
      end
      grad_params:zero()

      -- forward pass
      print({inputs})
      print({w2vinputs})
      os.exit()
      local outputs = model:forward({inputs,cudify(w2vinputs)})
      local err = lossfunc:forward(outputs, targets)
      err_sum = err_sum + err

      -- backward pass
      local grad_outputs = lossfunc:backward(outputs, targets)
      model:zeroGradParameters()
      model:backward({inputs,cudify(w2vinputs)},grad_outputs)

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
  pastalog('rnn-da', 'PPL (train)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ validation ------------------------
  model:evaluate() -- switch the model into eval mode

  -- init iterators
  local vtrainit = x_valid:subiter(opt.seqlen,opt.validsize)
  local vw2vit = w2v_valid:subiter(opt.seqlen,opt.validsize)

  local err_sum = 0
  for vtrainbatch,vw2vbatch in zip(vtrainit,vw2vit) do
    local i,inputs,targets = unpack(vtrainbatch)
    local _,w2vinputs,_ = unpack(vw2vbatch)

    targets = targetmodule:forward(targets)
    local outputs = model:forward({inputs,cudify(w2vinputs)})
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

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn-da', 'PPL (val)', ppl, epoch, 'http://localhost:8120/data')

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
print("Model saved: ..".. paths.concat(opt.savepath,opt.id..".t7"))
-- print command used to run main.lua file
run_command('\nCalled with: th main.lua ')

