-- A collection of various helper functions required to train the models and generate text.
local utils = {}

require "torch"
local stringx = require "pl.stringx"
local addOneDim = nn.utils.addSingletonDimension


-- make var accessible by utils.lua functions
function utils.register(var, varname)
  _G[varname] = var
end


-- DEBUG ONLY
function utils.px(...)
  print(...)
  os.exit()
end


-- print command line arguments used to start the script
function utils.run_command(args, message)
  io.write(message)
  local sorted = {}
  for k,v in ipairs(args) do
    table.insert(sorted,v)
  end
  for k,v in ipairs(sorted) do
    if k > 0 then
      io.write(string.format("%s ",v))
    end
  end
  io.write("\n")
end


function utils.zip(...)
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


-- convert tensors into cuda if set, (kept for interface compatibility)
function utils.cudify(vecs, w2v)
  if not w2v and opt.cuda then
    return vecs:squeeze(4):cuda()
  elseif not w2v and not opt.cuda then
    return vecs:squeeze(4):double()
  elseif w2v and opt.cuda then
    return vecs:cuda()
  elseif w2v and not opt.cuda then
    return vecs:double()
  end
end


-- convert tensors into cuda, use this function with sampler.lua
function utils.tocuda(vecs, w2v)
  if not w2v and opt.cuda then
    return vecs:cuda()
  elseif not w2v and not opt.cuda then
    return vecs:double()
  elseif w2v and opt.cuda then
    return vecs:cuda()
  elseif w2v and not opt.cuda then
    return vecs:double()
  end
end


function utils.clone(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in next, orig, nil do
      copy[utils.clone(orig_key)] = utils.clone(orig_value)
    end
    setmetatable(copy, utils.clone(getmetatable(orig)))
  else -- number, string, boolean, etc
    copy = orig
  end
  return copy
end


-- nullify given x input tensor
function utils.zerox(tensor, embdim)
  if opt.cuda then
    return torch.DoubleTensor(tensor:size()[1], tensor:size()[2], embdim):fill(0):cuda()
  else
    return torch.DoubleTensor(tensor:size()[1], tensor:size()[2], embdim):fill(0)
  end
end


-- nullify given tensor
function utils.zero(tensor)
  if opt.cuda then
    return torch.DoubleTensor(tensor:size()):fill(0):cuda()
  else
    return torch.DoubleTensor(tensor:size()):fill(0)
  end
end


-- join two tables together and return a new joined table
function utils.join(t1,t2)
  local newt = table.clone(t1)
  for i=1,#t2 do
    newt[#newt+1] = t2[i]
  end
  return newt
end


-- get the next word with max prob given the output probs
function utils.get_max_word(output)
  local p, idx = torch.max(torch.exp(output:squeeze(1)),1)
  return {ivocab[idx[1]],p}
end


-- get the next event token repr with max prob given the output probs
function utils.get_max_eword(output)
  local p, idx = torch.max(torch.exp(output:squeeze(1)),1)
  return {eivocab[idx[1]],p}
end


function utils.get_max_evec(output)
  -- LogSoftMax returns log-probability so you need to take an exp in order to get probs
  local max_p = {{},0}
  for i=1,output:size(1) do
    local p = torch.exp(output[i])
    if max_p[2] < p and ivocab2[i] ~= '<OOV>' then
      max_p[1] = ivocab2[i]
      max_p[2] = p
    end
  end
  local evec = utils.number2evec({tonumber(ivocab2[vocab2[max_p[1]]])})
  return evec,torch.IntTensor{vocab2[max_p[1]]}
end


-- sample the next word from multinomial distribution
function utils.get_mult_word(output)
  output:div(opt.temperature) -- scale by temperature
  local probs = torch.exp(output):squeeze()
  probs:div(torch.sum(probs)) -- renormalize so probs sum to one
  local prev_word = torch.multinomial(probs:float(),1) --:resize(1):float()
  local nword = {ivocab[torch.totable(prev_word)[1]],}
  local mprob = probs:float()[torch.multinomial(probs:float(),1)[1]]
  if opt.skipunk and nword[1] == "<unk>" then
    nword = utils.get_mult_word(output)
  end
  return nword,mprob
end


-- sample the next word from multinomial distribution
function utils.get_mult_eword(output)
  output:div(opt.temperature) -- scale by temperature
  local probs = torch.exp(output):squeeze()
  probs:div(torch.sum(probs)) -- renormalize so probs sum to one
  local prev_word = torch.multinomial(probs:float(),1) --:resize(1):float()
  local nword = {eivocab[torch.totable(prev_word)[1]],}
  local mprob = probs:float()[torch.multinomial(probs:float(),1)[1]]
  if opt.skipunk and nword[1] == "<unk>" then
    nword = utils.get_mult_word(output)
  end
  return nword,mprob
end


-- pretty print generated text
function utils.pprint(text_tbl)
  local newsent = true
  for _,w in pairs(text_tbl) do
    if w == "<eos>" then
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


function utils.evec2simple(events,maxdim)
  local vec = torch.IntTensor(maxdim):fill(0)
  local ecnt = 0
  for event_id in pairs(events) do
    if event_id ~= '174' then -- 174 is the no event id
      ecnt = ecnt + events[event_id]
    end
  end
  if ecnt ~= 0 then
    vec[{{1,ecnt}}] = 1
  end
  return vec
end


-- convert {3:1, 174:10} vector --> [5,1,1,1,1,1,1,1,1,1,1,0...0]
function utils.evec2big(events,weight,maxdim)
  local vec = torch.IntTensor(maxdim):fill(0)
  local esum, esum174 = 0, 0
  for event_id in pairs(events) do
    if event_id ~= '174' then -- 174 is the no event id
      esum = esum + events[event_id]
    else
      esum174 = esum174 + events[event_id]
    end
  end
  -- fill the event vector with event weight values
  if esum ~= 0 then
    vec[{{1,esum}}] = 5
  end
  -- fill the event vector with no event value 1
  if esum174 ~= 0 then
    vec[{{esum+1,esum+esum174}}] = 1
  end
  return vec
end


-- convert {3:1, -1:10} vector --> [5,1,1,1,1,1,1,1,1,1,1,0...0]
function utils.evec2full(events, weight, maxdim)
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


-- converts splitted line table into simple event vector
function event2simplevec(line_tab,maxdim)
  local events_tab = {}
  --for a,b in pairs(line_tab) do table.insert(events_tab,b) end
  for i=3,#line_tab,2 do
    events_tab[line_tab[i]] = line_tab[i+1]
  end
  return utils.evec2simple(events_tab,maxdim)
end


-- converts splitted line table into simple event vector
function event2fullvec(line_tab,weight,maxdim)
  local events_tab = {}
  --for a,b in pairs(line_tab) do table.insert(events_tab,b) end
  for i=3,#line_tab,2 do
    events_tab[line_tab[i]] = line_tab[i+1]
  end
  return utils.evec2full(events_tab,weight,maxdim)
end


-- converts topic number to 1-hot representation: 3 --> [0,0,1,0,0,0,0,0,0,0]
function utils.topic2vec(topic_id,dim)
  local tensor = torch.IntTensor(dim):fill(0)
  tensor[topic_id] = 1
  return tensor
end


-- convert IntTensor(91) [110000...0] --> IntTensor(1) [2]
function utils.evec2number(vec)
  local sum = torch.sum(vec)
  -- We can't use zero, Lookup table layer input must be 0 < 20 (#traindata.ivocab),
  -- depending on how LookupTable was initialized during training.
  if sum == 0 then
    return torch.IntTensor({18})
  end
  return torch.IntTensor({sum})
end


-- convert IntTensor(1) [2] --> IntTensor(91) [110000...0]
function utils.number2evec(tensor)
  local vec = torch.IntTensor(91):fill(0)
  -- 18 represents zero event vector
  -- if we plan to differenciate between event types we'll need to map 0 to some other number,
  -- or if we plan to introduce participant labels in addition to event labels.
  if tensor[1] == 18 then return vec end
  for i=1,tensor[1] do
    vec[i] = 1
  end
  return vec
end


-- add additional single dimension
function utils.add1(tensor)
  return addOneDim(tensor, 1)
end


-- convert {4 13} number to event vector
function utils.make_evector(evnum, noevnum) -- number of events, number of NO EVENT labels
  local evector = torch.IntTensor(91):fill(0)
  for i=1, evnum do
    evector[i] = 5 -- check what weight in dataloader.lua (5 or 10)
  end
  for i=1, noevnum do
    evector[evnum+i] = 1
  end
  return evector
end


-- convert labeled_tokens table to set topic-wise or globally, include blacklisted events
function utils.labeled2set(labeled_tokens, id, blacklisted) -- table, topic or global, blacklisted events table
  local topics = {["bath"] = 1,
                  ["bicycle"] = 2,
                  ["bus"] = 3,
                  ["cake"] = 4,
                  ["fligth"] = 5,
                  ["grocery"] = 6,
                  ["haircut"] = 7,
                  ["library"] = 8,
                  ["train"] = 9,
                  ["tree"] = 10}

  local topic =  topics[id] or 'global'
  local blacklisted =  blacklisted or {}
  if topic == 'global' then
    -- convert to set
    local labeled_set = {}
    for _,lmap in pairs(labeled_tokens) do
      for label,ltoks in pairs(lmap) do
        if not stringx.startswith(label, blacklisted) then
          for _, ltok in ipairs(ltoks) do
            labeled_set[ltok] = true
          end
        end
      end
    end
    return labeled_set
  else
    assert(labeled_tokens[topic], 'ERROR: no such topic name!')
    local labeled_set = {}
    for label, ltoks in pairs(labeled_tokens[topic]) do
      if not stringx.startswith(label, blacklisted) then
        for _, ltok in ipairs(ltoks) do
          labeled_set[ltok] = true
        end
      end
    end
    return labeled_set
  end
end


-- convert event vector into "2-24" event token representation
function utils.epred_vector2token(evec) -- event vector tensor
  if torch.sum(evec) == 0 then
    return string.format('19-%d', evec:size(1))
  end
  evs, noevs = 0, 0
  for i=1,evec:size(1) do
    if evec[i] == 5 then -- check what weight dataloader.lua uses (5 or 10)
      evs = evs + 1
    elseif evec[i] == 1 then
      noevs = noevs + 1
    end
  end
  return string.format('%d-%d', evs, noevs)
end


-- iterator, generates all possible next tokens as {prob,token} given a probs
function utils.iter_probs(probs,top)
  -- args: probs (table), top (int)
  function compare(a,b) return a[1] > b[1] end -- sort descending (max -> min)
  -- extract all probs and sort them in descending order (max first)
  local sorted_probs = {}
  for token_idx=1,probs:size(1) do -- iterate through prob distribution
    local p = torch.exp(probs[token_idx]) -- log probs to actual prob values
    table.insert(sorted_probs,{p,token_idx}) -- {1: {p, token_idx}s}
  end
  table.sort(sorted_probs,compare) -- sort by first value item
  -- create iterator
  local i = 0
  local n = top or #sorted_probs
  return function()
    i = i + 1
    if i <= n then return sorted_probs[i] end
  end
end


return utils