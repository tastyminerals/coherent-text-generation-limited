local ZipTable, parent = torch.class('ZipTable', 'nn.Container')

-- input : { {a1,a2}, {b1,b2}, {c1,c2} }
-- output : { {a1,b1,c1}, {a2,b2,c2} }
function ZipTable:__init()
   parent.__init(self)
   self.output = {}
   self.gradInput = {}
end

function ZipTable:updateOutput(inputTable)
   self.output = {}
   print('--------------------------- ZipTable ------------------------')
   print('>>> ZipTable INPUT')
   print({inputTable})
   for i,inTable in ipairs(inputTable) do
      for j,input in ipairs(inTable) do
         local output = self.output[j] or {}
         output[i] = input
         self.output[j] = output
      end
   end

   print('--------------------------- ZipTable ------------------------')
   print('>>> ZipTable OUTPUT')
   print({self.output})
   return self.output
end

function ZipTable:updateGradInput(inputTable, gradOutputTable)
   self.gradInput = {}
   --print('--------------------------- ZipTable backward ------------------------')
   --print('<<< INPUT')
   --print({inputTable})
   for i,gradOutTable in ipairs(gradOutputTable) do
      for j,gradOutput in ipairs(gradOutTable) do
         local gradInput = self.gradInput[j] or {}
         gradInput[i] = gradOutput
         self.gradInput[j] = gradInput
      end
   end
   --print('>>> OUTPUT')
   --print({self.gradInput})
   return self.gradInput
end
