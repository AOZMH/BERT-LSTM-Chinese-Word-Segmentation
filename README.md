# BERT-LSTM-Chinese-Word-Segmentation
BERT-LSTM-based Chinese word segmentation model on SIGHAN-2004

## Results
Tested on SIGHAN-2004 Chinese Word Segmentation dataset

|Measurements|Performance|
|:--------------|:----------:|
|TOTAL INSERTIONS|593|
|TOTAL DELETIONS|639|
|TOTAL SUBSTITUTIONS|1053|
|TOTAL NCHANGE|2285|
|OOV Rate|0.026|
|OOV Recall Rate|0.854|
|IV Recall Rate|0.988|
|TOTAL TRUE WORD COUNT|	106873|
|TOTAL TEST WORD COUNT|106827|
|**TOTAL TRUE WORDS RECALL**|**0.984**|
|**TOTAL TEST WORDS PRECISION**|**0.985**|
|**F MEASURE**|0.984|

## Execution
Train model:
> python main.py

Currently only support BERT-LSTM model.

Evluate:
> python eval.py

Will create a segmentated result on test data at /eval
