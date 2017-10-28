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
luarocks install json-lua
luarocks install luautf8-0.1.1-1.rockspec
luarocks install pastalog-scm-1.rockspec
luarocks install torchx-scm-1.rockspec
```

### Usage
There are seven models in this repository. 
In order to see available model options do `th main.lua --help`.
Each model configuration has been set to be optimal so each model will be initialized with predefined hyperparameters.
Simply start training the model:
```
th main.lua --cuda --adam 
```

You can experiment with `cutoff` parameter if the training becomes unstable:
```
th main.lua --cuda --adam --cutoff 10
```

After the model has been trained, you should run the text generation script `collect_gen.lua`.
This script will create 10 seed queries for the current model and generate 30 sentences per query:
```
th collect_gen.lua inscript_red_600_28ppl.t7 seedfile.json --silent > rnn-mod-da_samples.txt
```

If the generation process gets stuck (which might happen depending on the rnd initialization of `seedfile.json`).
You can recreate `seedfile.json` via `seedfile_gen.lua`, see available options via `th seedfile_gen.lua -h` (excluding **rnn**, **rnn-da**).

Finally, in order to evaluate sentence similarity for the generated samples you need to run `coherence.py` script.
```
python2 coherence.py results.txt 0
```

Where `results.txt` contains samples generated via `collect_gen.lua`.
Binary option `0` tells the script to calculate similarity scores without sentence length normalization (use `1` to normalize scores by sentence length).
The script outputs a cumulative sum table per each story.
 
### Results
Each model directory contains pretrained models and files with generated text.
The results reported in the paper are located in `evaluation/sims` directory.


