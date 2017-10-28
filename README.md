# coherent-text-generation-limited
Complemetary code for "Coherent Text Generation with Limited Training Data" master thesis.


### Installation
This **Torch** code is based on older **Element-Research/rnn** which has been recently deprecated in favor of **torch/rnn**.
Therefore, in case you've updated your **Torch** installation you'll explicitly need to install **Element-Research/rnn** afterwards:

```
git clone https://github.com/Element-Research/rnn.git
cd rnn
luarocks make rocks/rnn-scm-1.rockspec
``` 

You'll also need additional dependencies that might not be included with **Torch** distribution.

```
git clone https://github.com/tastyminerals/coherent-text-generation-limited.git
cd coherent-text-generation-limited
source ~/torch/install/bin/torch-activate
luarocks install luautf8-0.1.1-1.rockspec
luarocks install pastalog-scm-1.rockspec
luarocks install torchx-scm-1.rockspec
```

### Usage
There are seven models in this repository. 
In order to see available model options do `th main.lua --help`.
Each model configuration has been set to be optimal so each model will be initialized with preset hyperparameters.
Simply start training the model:
```
th main.lua --cuda --adam 
```

You can experiment with `cutoff` parameter if the training becomes unstable:
```
th main.lua --cuda --adam --cutoff 10
```




