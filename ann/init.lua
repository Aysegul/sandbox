require 'torch'
require 'xlua'
require 'nn'


-- create global ann table:
ann = {}

-- c lib:
require 'libann'

torch.include('ann', 'AddGaussianNoise.lua')
torch.include('ann', 'LeakyReLU.lua')
