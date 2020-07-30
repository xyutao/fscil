### How to use the index files for the experiments ?

The index files are named like "session_x.txt", where x indicates the session number. Each index file stores the indexes of the images that are selected for the session.
"session_1.txt" stores all the base class training images. Each "session_t.txt" (t>1) stores the 25 (5 classes and 5 shots per class) few-shot new class training images.
You may adopt the following steps to perform the experiments.

First, at session 1, train a base model using the images in session_1.txt;

Then, at session t (t>1), finetune the model trained at the previous session (t-1), only using the images in session_t.txt.

For evaluating the model at session t, first joint all the encountered test sets as a single test set. Then test the current model using all the test images and compute the recognition accuracy. 
