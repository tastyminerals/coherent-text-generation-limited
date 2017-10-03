local CAddTable, parent = torch.class('CAddTable', 'nn.Module')

function CAddTable:__init(ip)
  parent.__init(self)
  self.inplace = ip
  self.gradInput = {}
end

function CAddTable:updateOutput(input)
  print('--------- CAddTable forward --------')
  print('>>> CAddTable INPUT')
  print({input})
  --print('Checking self.output')
  --print({self.output})
  if self.inplace then
    self.output:set(input[1])
  else
    self.output:resizeAs(input[1]):copy(input[1])
  end
  for i=2,#input do
    self.output:add(input[i])
  end
  --print('<<< CAddTable OUTPUT')
  --print({self.output})
  return self.output
end

function CAddTable:updateGradInput(input, gradOutput)
  --print ('--------- CAddTable backward --------')
  --print('<<< INPUT')
  --print({input})
  --print('<<< gradOutput')
  --print({gradOutput})
  for i=1,#input do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    if self.inplace then
      self.gradInput[i]:set(gradOutput)
    else
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
    end
  end

  for i=#input+1, #self.gradInput do
    self.gradInput[i] = nil
  end
  --print('>>> OUTPUT')
  --print({self.gradInput})
  return self.gradInput
end
