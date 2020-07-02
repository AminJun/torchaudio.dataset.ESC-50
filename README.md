### ESC-50 
Here is an easy plug and play implementation to use [ESC-50](https://github.com/karolpiczak/ESC-50#download) dataset for audio tasks the same way you would use [torchaudio datasets](https://pytorch.org/audio/datasets.html#id3). 


### How to use
```
from ESC-50.py import ESC_50
train = ESC50(root='./data', download=True, train=True)
x,y = train[0]
```
