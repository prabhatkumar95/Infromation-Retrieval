Prabhat Kumar
MT17036
Information Retrival
Assignment 1

Files:
	1.create_dict.py : python script for creating Inverted Index for given Dataset
	2.main.py : python script containing Query functions
	3.json_dict.json : contains the inverted index in JSON format
	4.file_list.sav : contains data mapping file to User Defined file ID.

Assumptions:
	1.Files in the dataset have been mapped to User defined unqiue IDs
	2.Mapping have been stored in Pickle file 'file_list.sav'

Preprocessing:
	1.Headers from the files have been removed on the basis of first occurance of Empty Line
	2.Punctuations have been removed from the text files on the basis of set defined in "string.punctuation"
	3.Stop Words have been removed from the Resultant files.
	4.NLTK Tokenizer has been used to perform tokenization on the data files.
	5.After tokenization, Stop Words have been removed and Stemming have been performed using Porter Stemmer
