-- makes sure "data" dir contains files and their formatting is correct
require "paths"
require "lfs"
local stringx = require "pl.stringx"

local DIRSEP = package.config:sub(1,1) -- handle Windows or Unix
local DATAPATH = paths.cwd()..DIRSEP.."data"

function isdir(path)
  return lfs.attributes(path)["mode"] == "directory"
end

-- list all files in dir recursively
function get_files_recur(dir)
  local fpaths = fpaths or {}
  local function get_files(dir)
    local files = paths.dir(dir)
    for i=1,#files do
      if files[i] ~= '.' and files[i] ~= '..' then
        next_dir = dir..DIRSEP..files[i]
        if isdir(next_dir) then
          get_files(next_dir)
        else
          table.insert(fpaths,dir..DIRSEP..files[i])
        end
      end
    end
  end
  get_files(dir)
  return fpaths
end

function fread(fname)
  local ifile = assert(io.open(fname, 'r'))
  local fdata = ifile:read("*all")
  ifile:close()
  return fdata
end

function reader(arr)
  local data
  local i = 0
  local n = #arr
  return function()
    i = i + 1
    if i <= n then return fread(arr[i]) end
  end
end

function has_singletons(text)
  local vocab = {}
  for word in string.gmatch(text,"%S+") do
    if vocab[word] then
      vocab[word] = vocab[word] + 1
    else
      vocab[word] = 1
    end
  end
  local hassingle = false
  local singletons = {}
  for w,cnts in pairs(vocab) do
    if cnts == 1 then
      hassingle = true
      table.insert(singletons,w)
    end
  end
  if hassingle then
    -- uncomment to list all singletons
    --return singletons
    return true
  else
    return false
  end
end

-- extract all data from files
data = {}
fpaths = get_files_recur(DATAPATH)
iter = reader(fpaths)
for i=1,#fpaths do
  data[fpaths[i]] = iter()
end

describe("Test if data was preprocessed correctly.", function()
    test("Starting whitespace test: each sentence starts with a whitespace!",function()
        for fname,fdata in pairs(data) do
          local startspace = true
          if not fname:match(".csv") then -- do not check vec files
            for line in fdata:gmatch("(.-)\r?\n") do
              if not line:match("^%s.*") then
                startspace = {fname,line}
              end
            end
          end
          assert.is_true(startspace)
        end
      end)
   test("Lowercase test", function()
        for fname,fdata in pairs(data) do
          local upchars = false
          for w in string.gmatch(fdata, "%u") do
            upchars = fname
          end
          assert.is_false(upchars)
        end
      end)
   --[[ 
   test("Tokenization test: punctuation should be separated from words!",function()
        for fname,fdata in pairs(data) do
          local tailpunct = false
          if not fname:match(".csv") then -- do not check vec files
            for w in string.gmatch(fdata, "%S+") do
              if w:match("[%w>]+[%.?!;:,]+") then tailpunct = {fname,w} end
            end
          end
          assert.is_false(tailpunct)
        end
      end)]]
    --[[test("Singletons test: singleton words are not allowed!",function()
        local single = false
        for fname,fdata in pairs(data) do
          if fname:match("train.txt") then -- check only train data files
            local w = has_singletons(fdata)
            if w then single = {fname,w} end
          end
        end
        assert.is_false(single)
      end)]]
    test("Normalized digits test: digits should all be converted to single number!",function()
        local unique_nums = false
        for fname,fdata in pairs(data) do
          if not fname:match(".csv") then -- do not check vec files
            local num = fdata:match("%d")
            for n in fdata:gmatch("%d") do
              if num ~= n then unique_nums = {fname,n} end
            end
          end
        end
      end)
    test("Whitespace endings test: a line should end with a whitespace!",function()
        local endspace = true
        for fname,fdata in pairs(data) do
          if not fname:match(".csv") then -- do not check vec files
            for line in fdata:gmatch("(.-)\r?\n") do
              if not line:match("%s$") then
                endspace = {fname,line}
              end
            end
          end
        end
        assert.is_true(endspace)
      end)
  end)
