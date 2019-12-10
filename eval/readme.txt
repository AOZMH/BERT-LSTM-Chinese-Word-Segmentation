The script 'score' is used to generate compare two segmentations. The script takes three arguments:

1. The training set word list
2. The gold standard segmentation
3. The segmented test file

You must not mix character encodings when invoking the scoring script. For example:

% perl score gold/cityu_training_words.utf8 gold/cityu_test_gold.utf8 test_segmentation.utf8 > score.ut8