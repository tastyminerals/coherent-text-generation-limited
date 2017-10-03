--[[A small collection of lua snippets to work with multilingual text]]
-- to force reload the module use: package.loaded.tastytxt = nil; require 'tastytxt'
local tastytxt = {}

utf8 = require 'lua-utf8'
--pretty = require 'pl.pretty'

local DIRSEP = package.config:sub(1,1) -- handle Windows or Unix

-- split textual data into train, valid and test sets
function tastytxt.splitdata(fname, train, valid, test)
  assert(train + valid + test == 100)
  local bytes = tastytxt.size(fname,true)
  local trsize = tastytxt.round(bytes*(train/100))
  local vsize = tastytxt.round(bytes*(valid/100))
  local tsize = tastytxt.round(bytes*(test/100))
  print('total: '..bytes)
  print('train: '..trsize)
  print('valid: '..vsize)
  print('test: '..tsize)

  local f = io.open(fname,'r')
  local data = {tr={},v={},t={}}
  local c
  local set = "tr"
  repeat
    c = f:read(1)
    table.insert(data[set],c)
    if #data['v'] == 0 and trsize <= #data["tr"] and c == ' ' then
      set = 'v'
    elseif #data['t'] == 0 and #data['v'] > 0 and vsize <= #data['v'] and c == ' ' then
      set = 't'
    end
  until c == nil
  tastytxt.write(fname..'_train.txt',table.concat(data["tr"]))
  tastytxt.write(fname..'_valid.txt',table.concat(data['v']))
  tastytxt.write(fname..'_test.txt',table.concat(data['t']))
end

function tastytxt.round(num, decimal)
  local mult = 10^(decimal or 0)
  return math.floor(num * mult + 0.5) / mult
end

function tastytxt.size(file,usebytes)
  local f = io.open(file,'r')
  local current = f:seek()      -- get current position
  local size = f:seek("end")    -- get file size
  f:seek("set",current)        -- restore position
  f:close()
  if not usebytes then
    return tastytxt.round(size/1024/1024,2)
  else
    return size
  end
end

-- if a dir contains files, return an array of file full paths
function tastytxt.listdir(path)
    local i,files = 0,{}
    local pfile = io.popen('ls "'..path..'"')
    for fname in pfile:lines() do
        i = i + 1
        fpath = path..DIRSEP..fname
        files[i] = fpath
    end
    pfile:close()
    return files
end

-- read corpus file
function tastytxt.read(fname)
  local ifile = assert(io.open(fname, 'r'))
  local fdata = ifile:read("*all")
  ifile:close()
  return fdata
end

-- write text to file, if table is given, concatenate into text
function tastytxt.write(fname,text)
  local text = text
  local ofile = fname
  -- handle table
  if type(text) == "table" then
    text = table.concat(text, '\n')
  end
  -- write file
  ofile = assert(io.open(fname,'w'))
  ofile:write(text)
  ofile:close()
  return true
end

-- lowercase all text in a file
function tastytxt.low(text)
  local text = text
  return utf8.lower(text)
end

-- separate only sentence delimiters from text.
function tastytxt.soft_tokenize(text)
  local text = text
  local lns = {}
  -- iterate over each line, otherwise we are breaking text formatting
  for line in utf8.gmatch(text,"[^\n]+") do
    -- surround punctuation with spaces
    line = utf8.gsub(line,"([.,!?]+)"," %1 ")
    -- replace mult space with single space
    line = utf8.gsub(line,"%s%s+"," ")
    -- fix auxiliary contractions
    table.insert(lns,line)
  end
  return table.concat(lns,'\n')
end


-- separate punctuation from text
function tastytxt.tokenize(text)
  local text = text
  local lns = {}
  -- iterate over each line, otherwise we are breaking text formatting
  for line in utf8.gmatch(text,"[^\n]+") do
    -- surround punctuation with spaces
    line = utf8.gsub(line,"(%p)"," %1 ")
    -- replace mult space with single space
    line = utf8.gsub(line,"%s%s+"," ")
    -- fix auxiliary contractions
    line = utf8.gsub(line," ' s "," 's ") -- for English
    line = utf8.gsub(line,"n ' t "," n't ") -- for English
    line = utf8.gsub(line," ' t "," 't ") -- for English
    line = utf8.gsub(line," ' ve "," 've ") -- for English
    line = utf8.gsub(line," ' m "," 'm ") -- for English
    line = utf8.gsub(line," ' re "," 're ") -- for English
    line = utf8.gsub(line," ' d "," 'd ") -- for English
    line = utf8.gsub(line," ' ll "," 'll ") -- for English
    table.insert(lns,line)
  end
  return table.concat(lns,'\n')
end

-- count the number of tokens using str match
function tastytxt.cnt(text)
  local text = text
  -- separate punctuation from words first
  local _,cnt = utf8.gsub(tastytxt.tokenize(text),"%S+",'')
  return cnt
end

-- count only punctuation chars
function tastytxt.cnt_punk(text)
  local text = text
  -- separate punctuation from words first
  local _,cnt = utf8.gsub(tastytxt.tokenize(text),"%p",'')
  return cnt
end

-- strip punctuation
function tastytxt.stripp(text)
  local text = text
  local lns = {}
  -- iterate over each line, otherwise we are breaking text formatting
  for line in utf8.gmatch(text,"[^\n]+") do
    line = utf8.gsub(line,"%p",' ')
    -- replace mult space with single space
    line = utf8.gsub(line,"%s%s+"," ")
    table.insert(lns,line)
  end
  return table.concat(lns,'\n')
end

-- refit corpus so that each sentence takes one line, whitespace follows sentence delimiter
-- regex involved, use carefully
function tastytxt.sent2line(text)
  local sents = {}
  local lns = tastytxt.splitlines(text)
  for _,line in ipairs(lns) do
    for sent in utf8.gmatch(line, "[^%.%?%!]+[%.%!%?]+ ?") do
      table.insert(sents,sent)
    end
  end
  return sents
end


-- tokenize text and return lines array of tokens array
function tastytxt.lines(text)
  local lns = {}
  local toks = {}
  -- split by lines
  for line in utf8.gmatch(text,"[^\n]+") do
    toks = {}
    -- split each line by tokens
    for word in utf8.gmatch(line,"%S+") do
      table.insert(toks,word)
    end
    table.insert(lns,toks)
  end
  return lns
end

-- count tokens and return {token=cnt} map
function tastytxt.vocab(text)
  local vocab = {}
  -- tokenize punctuation first
  text = tastytxt.tokenize(text)
  -- count tokens
  for word in utf8.gmatch(text,"%S+") do
    if vocab[word] then
      vocab[word] = vocab[word] + 1
    else
      vocab[word] = 1
    end
  end
  return vocab
end

-- return {tokens} occurring <= max number of times
function tastytxt.rare(text,max)
  -- count tokens
  local wordmap = tastytxt.vocab(text)
  -- find singletons
  local singletons = {}
  for k,v in next,wordmap,nil do
    if v <= max then
      table.insert(singletons,k)
    end
  end
  --pretty.dump(singletons)
  return singletons
end

-- return a set of text tokens
function tastytxt.unique(text)
  local tokset = {}
  local cnt = 0
  -- tokenize punctuation first
  local text = tastytxt.tokenize(text)
  for token in utf8.gmatch(text,"%S+") do
    if tokset[token] == nil then
      tokset[token] = true
      cnt = cnt + 1
    end
  end
  return tokset,cnt
end

-- replace a digit with 0
function tastytxt.dig2zero(text)
  local text = text
  return utf8.gsub(text,"%d+","0")
end

-- replace cnt-frequency tokens with <unk>
function tastytxt.repunk(text,cnt)
  local inner = {}
  local outer = {}
  -- find rare tokens
  local rare = tastytxt.rare(text,cnt or 1)
  -- tokenize text and return tokens array
  local text_arr = tastytxt.lines(text)
  -- replace rare tokens with <unk>
  for l=1,#text_arr do -- iterating lines {}
    for t=1,#text_arr[l] do -- iterating tokens {}
      for r=1,#rare do
        if text_arr[l][t] == rare[r] then text_arr[l][t] = "<unk>" end
      end
    end
  end
  -- concat inner level
  for i=1,#text_arr do
    table.insert(inner,table.concat(text_arr[i],' '))
  end
  -- concat outer level
  for i=1,#inner do
    table.insert(outer,inner[i])
  end
  return table.concat(outer, "\n")
end

--[[Do complete preprocessing of a corpus which includes:
lowecasing, tokenization, replacing rare (singleton) tokens with <unk>, replacing
digits with zeros. Include an option to include/exclude punctuation and <unk>.

nopunct -- strip punctuation if true
unk -- search and replace singletons with <unk> label if true
]]
function tastytxt.prep(text,nopunct,unk)
  -- lowercase text
  print('> lowercasing...')
  local text = tastytxt.low(text)
  if nopunct then
    print('> stripping punctuation...')
    text = tastytxt.stripp(text)
  end
  -- tokenize punctuation
  print('> tokenizing punctuation...')
  text = tastytxt.tokenize(text)
  -- replace digits with zeros
  print('> digits to zero...')
  text = tastytxt.dig2zero(text)
  -- replace singletons with <unk>
  collectgarbage()
  if unk then
    print('> adding <unk>...')
    text = tastytxt.repunk(text,1)
  end
  return text
end

-- if a table contains only str keys {token:cnt}, count the number of str keys
function tastytxt.countstrkeys(table)
    local keycnt = 0
    for key,val in next,table do
        keycnt = keycnt + 1
    end
    return keycnt
end

-- split text into tokens using separator or whitespace
function tastytxt.split(text,sep)
  if sep then
    assert(type(sep) == "string" and #sep == 1, "Incorrect separator!")
  end

  local toks = {}
  -- split into lines first
  for _,line in ipairs(tastytxt.splitlines(text)) do
    for word in utf8.gmatch(line,string.format("[^%s]+",sep) or "%S+") do
      table.insert(toks,word)
    end
  end
  return toks
end

-- split text into table of lines
function tastytxt.splitlines(text)
  local lns = {}
  -- split into lines
  for line in utf8.gmatch(text,"[^\n]+") do
    table.insert(lns,line)
  end
  return lns
end

-- remove only the first sentence from each line of the given text
function tastytxt.remove_first_sentence(text)
  local lns = tastytxt.splitlines(text)
  local new_lns = {}
  for i=1,#lns do
    -- notice the whitespace after the dot
    -- here gsub returns 2 params and overloaded table.insert that accepts 3 values is invoked
    table.insert(new_lns, (utf8.gsub(lns[i],"^[^%.]+%. ","",1)))
  end
  return new_lns
end

-- count the number of elements in a table
function tastytxt.tabcnt(tab)
  local cnt = 0
  for _ in pairs(tab) do
    cnt = cnt + 1
  end
  return cnt
end

-- iterate over two sequences in parallel, shortest sequence ends the loop
function tastytxt.zip(...)
  local arrays, ans = {...}, {}
  local index = 0
  return function()
      index = index + 1
      for i,t in ipairs(arrays) do
        if type(t) == 'function' then
          ans[i] = t()
        else
          ans[i] = t[index]
        end
        if ans[i] == nil then return end
      end
      return unpack(ans)
    end
end


return tastytxt
