# recsys_final_project

``` shorten_data.py ``` reads the original twitter data, and randomly subsamples 0.0625 part of it, saving into "train_short.tsv" and "val_short.tsv" files.

``` process.py ``` reads the "train_short.tsv" and "val_short.tsv" file, extracts users features and creates one label column with 5 different classes of interactions (including "no engagement"), and saves it as a tab-separated "class_dataframe_train.tsv" and "class_dataframe_val.tsv" files.

``` run_random_forest.py ``` reads "class_dataframe_train.tsv", runs sklearn RandomForestClassifier with gridsearched separately parameters, and evaluates on the "class_dataframe_val.tsv" file
