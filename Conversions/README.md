torch-demo-tools
================


#Conversions

## convertLinearmodelConvMM  
   on CPU run as 
   ```lua
       th convertLinearmodelConvMM.lua './dir/model.net' 
   ```

to modify your trained network with Linear layer. By replacing the linear layers with ConvolutionMM, you can feed the network bigger inputs than the ones network trained with. You need to modify the input tensor based on your training. 

