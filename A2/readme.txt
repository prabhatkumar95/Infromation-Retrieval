Prabhat Kumar
MT17036

13th Feb 2018

IR Assignment 2

The project contains the following files:

    1. preprocess.py
        Contains source code to primarily create tf-idf matrix(Words,Files)
    2. title_extractor.py
        Extracts word count and Title for respective files and saves to title_extract.json
    3. main.py
        Contains source code for TF_IDF match score as well as for cosine similarity
    4. tf_dict.sav
        Created by preprocess.py, contains dictionary with Keys corresponding to Words and List to each value with each
        value denoting TF-IDF value for the given file.
    5. title_extract
        Contains dictionary, with key as file name and a list containing preprocessed tokens of title as well as word
        count
    6. idf_val
        Contains idf value of each word, with respect to given document set.
    7. results.sav
        Used for caching results of last 20 unique queries made.