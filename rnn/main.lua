--[[
This is vanilla GRU lm adapted from:
  https://github.com/Element-Research/rnn/blob/master/examples/recurrent-language-model.lua

Misc training configurations:
  th main.lua --cuda --device 2 --progress --cutoff 4 --seqlen 10
  th main.lua --progress --cuda --seqlen 20 --hiddensize '{200,200}' --batchsize 20 --startlr 1 --cutoff 5 --maxepoch 13 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'
  th main.lua --progress --cuda --seqlen 35 --uniform 0.04 --hiddensize '{1500,1500}' --batchsize 20 --startlr 1 --cutoff 10 --maxepoch 50 --schedule '{[15]=0.87,[16]=0.76,[17]=0.66,[18]=0.54,[19]=0.43,[20]=0.32,[21]=0.21,[22]=0.10}' --dropout 0.65

]]
require "rnn"
require "paths"
require "optim"
local loader = require "loader.dataloader"
local pastalog = require 'pastalog'


------------------------ helper functions ------------------------
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
cmd:option("--saturate",100,"epoch at which linear decay of lr reaches minlr") -- set 400 if not adam
cmd:option("--schedule",'',"learning rate schedule, e.g. '{[5]=0.004,[6]=0.001}'")
cmd:option("--momentum",0.9,"prevents the network from converging to local minimum")
cmd:option('--adam', false, 'use ADAM instead of SGD as optimizer')
cmd:option('--adamconfig', '{0, 0.999}', 'ADAM hyperparameters beta1 and beta2')
cmd:option("--maxnormout",-1,"prevents overfitting, max 12-norm of each layer's output neuron weights")
cmd:option("--cutoff",10,"max 12-norm of concatenation of all gradParam tensors")
cmd:option("--maxepoch",200,"max number of epochs to run") -- 1000
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


------------------------ process command line arguments ----------------------------
local opt = cmd:parse(arg or {})
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
local traindata,validdata,testdata = loader.load(opt.path,{opt.batchsize,1,1})

if not opt.silent then
  print("Vocab size: "..#traindata.ivocab)
  print(string.format("Train data is split into %d sequences of total %d length",opt.batchsize,traindata:size()))
end


------------------------ create network ------------------------
local model = nn.Sequential()

-- build input layer as lookup table
local lookup = nn.LookupTable(#traindata.ivocab,opt.inputsize) -- (17134x200)
lookup.maxNorm = opt.maxnormout -- prevents weird maxnormout behaviour
model:add(lookup) -- input layer has dimensions (seqlen x batchsize)

-- gru has dropout, add if rnn was selected
if opt.rnn then model:add(nn.Dropout(opt.dropout)) end

-- nn.SplitTable creates a table with specific fields like "dimension", "gradInput", "output", etc.
-- where "dimension" contains given data
model:add(nn.SplitTable(1))

-- build recurrent layers
local stepmodule = nn.Sequential() -- a module which is applied as each time t step
local inputsize = opt.inputsize

-- since we can supply more than one rnn layer, we do a loop
for i,hiddensize in ipairs(opt.hiddensize) do
  local rnn
  if not opt.rnn then
    rnn = nn.GRU(inputsize,hiddensize,nil,opt.dropout/2)
  else
    local rm = nn.Sequential() -- input is {x[t],h[t-1]}
    local pt = nn.ParallelTable()
    pt:add(i==1 and nn.Identity() or nn.Linear(inputsize,hiddensize)) -- input layer
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
stepmodule:add(nn.Linear(inputsize,#traindata.ivocab))
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
xplog.vocab = traindata.vocab
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
opt.trainsize = opt.trainsize == -1 and traindata:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validdata:size() or opt.validsize

local begin_time = os.date()
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print()
  print("Epoch #"..epoch)

  sgdconfig = {
    learningRate = opt.lr,
    momentum = opt.momentum
    }

  local timer = torch.Timer()
  model:training() -- switch the model into training mode
  local err_sum = 0
  for i,inputs_batch,targets_batch in traindata:subiter(opt.seqlen,opt.trainsize) do
    -- here targets get forwarded through targetmodule which is nn.SplitTable(1)
    -- after nn.SplitTable targets is a table of 5 IntTensors of size 32
    local targets = targetmodule:forward(targets_batch)  -- targets_batch (IntTensor 5x32)

    local inputs = inputs_batch

    local function feval(val)
      if val ~= params then
        params:copy(val)
      end
      grad_params:zero()

      -- forward pass
      local outputs = model:forward(inputs)

      local err = lossfunc:forward(outputs, targets)
      err_sum = err_sum + err

      -- backward pass
      local grad_outputs = lossfunc:backward(outputs, targets)
      model:zeroGradParameters()
      model:backward(inputs, grad_outputs)

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
    if opt.meanNorm then
      print("mean grad_param norm",opt.meanNorm)
    end
  end

  if cutorch then cutorch.synchronize() end
  local speed = timer:time().real/opt.trainsize
  print(string.format("Elapsed time: %f",timer:time().real))
  print(string.format("Speed: %f sec/batch",speed))

  local ppl = torch.exp(err_sum/opt.trainsize)
  print("Training PPL before validation: "..ppl)
  xplog.trainppl[epoch] = ppl

  -- API is pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn', 'PPL (train)', ppl, epoch, 'http://localhost:8120/data')


  ------------------------ validation ------------------------
  model:evaluate() -- switch the model into eval mode
  local err_sum = 0

  for i,inputs,targets in validdata:subiter(opt.seqlen,opt.validsize) do
    targets = targetmodule:forward(targets)

    local outputs = model:forward(inputs)
    local err = lossfunc:forward(outputs,targets)
    err_sum = err_sum + err
  end

  -- Perplexity = exp(sum(NLL)/#2)
  local ppl = torch.exp(err_sum/opt.validsize)
  print("Training PPL : "..ppl)
  xplog.valppl[epoch] = ppl
  ntrial = ntrial + 1

  -- report via pastalog(modelName, seriesName, value, step, [url])
  pastalog('rnn', 'PPL (valid)', ppl, epoch, 'http://localhost:8120/data')

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
