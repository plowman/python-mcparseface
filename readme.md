## What this project does

This project lets you use Google's SyntaxNet Google's SyntaxNet (AKA [Parsey McParseface](https://github.com/tensorflow/models/tree/master/syntaxnet)) 
from python code and without hours of installing and training models.  


## How to run the parser

```
python main.py 
```

## How to install steps (OSX)

bazel:
  * brew install bazel
    * versions 0.2.0 - 0.2.2b, NOT 0.2.3

swig:
  * brew install swig

protocol buffers, with a version supported by TensorFlow:
  * pip install -U protobuf==3.0.0b2

asciitree, to draw parse trees on the console for the demo:
  * pip install asciitree

numpy, package for scientific computing:
  * pip install numpy


## Install steps (Windows)

* I am not good with windows. If you have been able to make this work on Windows, I would gladly merge a Pull Request from you.

