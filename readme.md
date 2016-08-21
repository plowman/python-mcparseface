## What this project does

This project lets you use Google's SyntaxNet (AKA [Parsey McParseface](https://github.com/tensorflow/models/tree/master/syntaxnet)) 
from python code and without hours of installing and training models.  


## How to run the parser

```python
import pyparseface

parsed_sentence_dict = pyparseface.parse_sentence(
  "Maybe there was once a human who looked like you, and somewhere along the "
  "line you killed him and took his place. And your superiors don't know.")
print("OrderedDict: %s\n" % parsed_sentence_dict)
pyparseface.pretty_print_dict(parsed_sentence_dict)
```

#### Output
```
OrderedDict: OrderedDict([(u'was VBD ROOT', OrderedDict([(u'Maybe RB advmod', OrderedDict()), (u'there EX expl', OrderedDict()), (u'once RB advmod', OrderedDict()), (u'human NN nsubj', OrderedDict([(u'a DT det', OrderedDict()), (u'looked VBD rcmod', OrderedDict([(u'who WP nsubj', OrderedDict()), (u'like IN prep', OrderedDict([(u'you PRP pobj', OrderedDict())]))]))])), (u', , punct', OrderedDict()), (u'and CC cc', OrderedDict()), (u'killed VBD conj', OrderedDict([(u'somewhere RB advmod', OrderedDict([(u'along IN prep', OrderedDict([(u'line NN pobj', OrderedDict([(u'the DT det', OrderedDict())]))]))])), (u'you PRP nsubj', OrderedDict()), (u'him PRP dobj', OrderedDict()), (u'and CC cc', OrderedDict()), (u'took VBD conj', OrderedDict([(u'place. NN dobj', OrderedDict([(u'his PRP$ poss', OrderedDict())]))])), (u'And CC cc', OrderedDict()), (u'know VB conj', OrderedDict([(u'superiors NNS nsubj', OrderedDict([(u'your PRP$ poss', OrderedDict())])), (u'do VBP aux', OrderedDict()), (u"n't RB neg", OrderedDict())]))])), (u'. . punct', OrderedDict())]))])

was VBD ROOT
 +-- Maybe RB advmod
 +-- there EX expl
 +-- once RB advmod
 +-- human NN nsubj
 |   +-- a DT det
 |   +-- looked VBD rcmod
 |       +-- who WP nsubj
 |       +-- like IN prep
 |           +-- you PRP pobj
 +-- , , punct
 +-- and CC cc
 +-- killed VBD conj
 |   +-- somewhere RB advmod
 |   |   +-- along IN prep
 |   |       +-- line NN pobj
 |   |           +-- the DT det
 |   +-- you PRP nsubj
 |   +-- him PRP dobj
 |   +-- and CC cc
 |   +-- took VBD conj
 |   |   +-- place. NN dobj
 |   |       +-- his PRP$ poss
 |   +-- And CC cc
 |   +-- know VB conj
 |       +-- superiors NNS nsubj
 |       |   +-- your PRP$ poss
 |       +-- do VBP aux
 |       +-- n't RB neg
 +-- . . punct
```

## Author

Most of the hard stuff was written by Google. [Ben Plowman](https://twitter.com/benplowman) made it slightly easier to get up and running.
