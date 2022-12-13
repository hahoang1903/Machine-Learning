# INT3405E_41 Group Project - PhiGroup

### Foody Sentiment Analysis using PhoBERT
The project fine-tune a pre-trained PhoBERT for classifying reviews on Foody.vn. Code was written as a notebook to run on Kaggle notebook system. It can also run on Google Colab with some small changes (path, and mount drive)

### Trained model
The trained model's state was saved in *model.pth*. You can load model state using
```
from transformers import AutoModel 
import torch

model = AutoModel.from_pretrained('vinai/phobert-base')
model.load_state_dict(torch.load(path/to/model.pth))
```

Inputs to model **MUST** already be word-segmented. We use VnCoreNLP for word-segment and saved the model in *vncorenlp* folder. Here you can load the word-segmenter from **VnCoreNLP**
```
import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir='/path/to/vncorenlp', max_heap_size='-Xmx500m')
```

and segment words using
```
word_segmented = rdrsegmenter.word_segment('Tôi học ở đại học Công Nghệ')
```

then you can use **PhoBERT** tokenizer to make input for model
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
tokenizer.encode(word_segmented)
```

### Results
Trained model achieved **99.73%** on train dataset, and **89.98%** on public test dataset
