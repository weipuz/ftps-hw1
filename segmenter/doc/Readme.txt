Group FTPS
23/09/2014

Program brief description
------------------------------
For this word segmentation program, we developed a segmenter   based on the pseudo-code combined with a mixture of unigram and  bigramd model in which added additive smoothing.
------------------------------------

Program specific description based on code
----------------------------------
-There are three basic calss named Pdist,WordDict and Segmenter in the baseline.py

-The purpose of the Pdist is to get the probability distribution estimated from couts in datafile as described in the example code

-In WordDict class, 
	1.In load function,it loads both unigram dictionary and 	bigram dictionary. To uniform the format as unigram 	dictionary,we combined two separate words into one word 	among the bigram dictionary. 
	2.In __call__ funtion, we add additive smoothing to 	deallocate the probability of each words.

-In Segmenter class, 
	1.The __init__ function load the input file and previous 	combined dictionary. 

	2.In the segmentSent function,we followed the pseudo-code to 	segment each sentence.

	3.Theoretically, a continuing of single and rare words is 	always a proper 	noun phrase.To improve the accuracy, the 	combineSingle function help to deal with single and rare 	word. Specifically, we set a threshold as 8.0e-	5,once 	probability of a word is smaller than this 	threshold, 	it is defined as rare word. We store 	it and compare it 	to its previous and next words. Once its previous and 	next 	words are rare, than we combine both of them. 
------------------------------------