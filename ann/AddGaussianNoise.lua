local AddGaussianNoise, parent = torch.class('nn.AddGaussianNoise', 'nn.Module')

function AddGaussianNoise:__init(std, ip)
  parent.__init(self)
  assert(type(std) == 'number', 'std is not scalar!')
  self.std = std or 0.1
  self.train = true
 
  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end

   self.gaussian = torch.Tensor()
end

function AddGaussianNoise:updateOutput(input)
  if self.inplace then
    self.output = input
  else
    self.output:resizeAs(input):copy(input)
  end

  if self.train then
     self.gaussian:resizeAs(input)
     self.gaussian:normal(0,1):mul(self.std)
     self.output:add(self.gaussian)
  end

  return self.output

end

function AddGaussianNoise:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput = gradOutput
      -- restore previous input value
      input:add(-self.gaussian)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end

   return self.gradInput
end



