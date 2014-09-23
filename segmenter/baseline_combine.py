
import sys, codecs, optparse, os, math
from nltk.probability import FreqDist
import nltk

reload(sys) 
sys.setdefaultencoding('UTF8')

# parse the input command lines
optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0 
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)

# word dictionary with universal interfaces
class WordDict(dict):
	"dictionary of Chinese words and their counts"

	# initialize by input file
	def __init__(self, filenames=["data/count_1w.txt"]):
		self.maxlen = 0
		self.total = 0
		# load each file, and combine them
		for name in filenames:
			self.load(name)

	# call function as dictionary
	def __call__(self, word, word1 = None, word2 = None, word3 = None):
		all_words = word
		# compatible with 2, 3, 4-word model
		if None != word1:
			all_words += word1
		if None != word2:
			all_words += word2
		if None != word3:
			all_words += word3
		# word exists
		if all_words in self:
			# direct probability
			prob = float(self[all_words]["count"]) / float(self.total)
			# smoothing
			if len( self[all_words]["words"] ) > 1:
				first_prob = self( self[all_words]["words"][0] )
				if None != first_prob:
					# Additive Smoothing (slides)
					delta = 0.5
					prob = (delta + prob) / (self.total + first_prob)
					# smooth like discussion/topic/using-bigrams-with-the-iterative-algorithm/
					k = 0.7
					prob = k * prob + (1.0 - k) * first_prob
			return prob
		# not exist, single character
		elif 1 == len(all_words):
			return 1.0 / self.total
		# does not exist
		else:
			return None

	# load dictionary from file
	def load(self, filename):
		for line in file(filename):
			(key, freq) = line.split("\t")
			try:
				utf8key = unicode(key, 'utf-8')
			except:
				raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
			# all words
			words = utf8key.split(" ")
			count = self.get(utf8key, 0) + int(freq)
			self.total += count
			self.maxlen = max(len(utf8key), self.maxlen)
			all_words = "".join(words)
			# if not exist, or better
			if not (all_words in self) or count > self[all_words]["count"] or len(words) < len(self[all_words]["words"]) :
				# add to dictionary
				self[all_words] = {"key": utf8key, "words": words, "count": count}



class Segmenter():
	"segmenter"

	# load a input file and dictionary file
	def __init__(self, file, dict_paths = [opts.counts1w]):
		self.text = [ unicode(text.strip(), "utf-8") for text in open(file) ]
		#self.test_file = codecs.open("test", "w", "utf-8")
		self.test_file = open("test.log", "w")
		#self.output_file = codecs.open("output", "w", "utf-8")
		self.output_file = open("output.log", "w")
		# create a dictionary
		self.dict = WordDict(dict_paths)

	# destructor
	def __exit__(self, type, value, traceback):
		self.test_file.close()

	# print the test comments
	def printTest(self, output):
		self.test_file.write(output + unicode("\n",'utf-8'))

	# print output to file
	def printOutput(self, output):
		self.output_file.write(output + unicode("\n",'utf-8'))

	def min(self, num0, num1):
		if num0 < num1:
			return num0
		else:
			return num1

	# convert entry to string for printing
	def strEntry(self, entry):
		s = "chartEntry(" + entry["word"] + ", start=" + repr(entry["start"]) + ", end=, logprob=" + repr(entry["logprob"]) + ", backptr="
		if None != entry["back"]:
			s += entry["back"]["word"]
		else:
			s += "None"
		return s

	# pop the top of a heap
	def heapPop(self, heap):
		top = {"word": "", "start": sys.maxint, "logprob": 0.0, "back": None}
		top_index = 0;
		for index, item in enumerate(heap):
			if (item["start"] < top["start"]) or (item["start"] == top["start"] and item["logprob"] > top["logprob"]):
				top = item
				top_index = index
		if top["word"] != "":
			del heap[top_index]
			return top
		else:
			return None

	# segment the entire input file
	def run(self):
		ans = []
		for sentence in self.text:
			ans0 = self.segmentSent(sentence)
			self.printOutput(ans0)
			ans.append(ans0)
		return ans

	# segment a sentence
	def segmentSent(self, sentence):
		# sentence: the input sequence of characters
		if 0 == len(sentence):
			return ""
		self.printTest("input: " + sentence)
		# chart: the dynamic programming table to store the argmax for every prefix of input, indexed by character position in input
		chart = [None] * (len(sentence) + 1)
		# heap: a list or priority queue containing the entries to be expanded, sorted on start-position or log-probability
		heap = []
		# entry: each entry in the chart has four components: Entry(word, start-position, log-probability, back-pointer), the back-pointer in each entry links it to a previous entry that it extends
		# change to Entry(start-position, log-probability, word, back-pointer)

		## Initialize the heap ##

		# for each word that matches input at position 0
		for i in range(1, self.min(len(sentence) + 1, self.dict.maxlen)):
			word = sentence[: i]
			p = self.dict(word)
			if p != None:
				# insert Entry(word, 0, logPw(word), None) into heap
				entry = {"word": word, "start": 0, "logprob": math.log10(p), "back": None} #(0, math.log10(p), word, None)
				#self.printTest("Adding: " + entry["word"] + " " + repr(entry["logprob"]) )
				heap.append(entry)

		## Iteratively fill in chart[i] for all i ##

		# while heap is nonempty:
		while len(heap):
			# entry = top entry in the heap
			entry = self.heapPop(heap)
			if None == entry:
				break
			# get the endindex based on the length of the word in entry
			end_index = entry["start"] + len(entry["word"]) - 1
			#self.printTest( "pop: endindex=" + repr(end_index) + " word=" + entry["word"] + " logprob=" + repr(entry["logprob"]) )
			# if chart[endindex] has a previous entry, preventry
			if end_index < len(chart) and None != chart[end_index] and None != chart[end_index]["back"]:
				prev_entry = chart[end_index]["back"]
				# if entry has a higher probability than preventry:
				if entry["logprob"] > prev_entry["logprob"]:
					# chart[endindex] = entry
					chart[end_index] = entry
				# if entry has a lower or equal probability than preventry:
				else:
					# we have already found a good segmentation until endindex
					continue
			else:
				# chart[endindex] = entry
				chart[end_index] = entry
			# for each newword that matches input starting at position endindex+1
			for i in range(end_index + 2, len(sentence) + 1):
				new_word = sentence[end_index + 1 : i]
				p = self.dict(new_word)
				if None != p:
					# newentry = Entry(newword, endindex+1, entry.log-probability + logPw(newword), entry)
					new_entry = {"word": new_word, "start": end_index + 1, "logprob": entry["logprob"] + math.log10(p), "back": entry} #(end_index + 1, entry[1] + math.log10(p), new_word, entry)
					# if newentry does not exist in heap:
					if not(new_entry in heap):
						#self.printTest("endIndex= " + repr(end_index) + " : newEntry= " + self.strEntry(new_entry))
						# insert newentry into heap
						heap.append(new_entry)

		## Get the best segmentation ##

		# finalindex is the length of input
		final_index = len(sentence) - 1
		# finalentry = chart[finalindex]
		final_entry = chart[final_index]
		seg = []
		# The best segmentation starts from finalentry and follows the back-pointer recursively until the first word
		while None != final_entry:
			self.printTest("final[ 0 ]: " + self.strEntry(final_entry))
			seg.append(final_entry["word"])
			final_entry = final_entry["back"]
		# reverse it
		seg = seg[: : -1]

		seg = self.separateMultiple(seg)
		seg = self.combineSingle(seg)
		#seg = self.restoreMissing(seg, sentence)

		ans = " ".join(seg)
		self.printTest(ans + "\n")
		return ans

	# separate multiple words in an item in a dictionary
	def separateMultiple(self, seg):
		index = 0
		while index < len(seg):
			word = seg[index]
			if word in self.dict:
				seg[index] = self.dict[word]["words"][0]
				for i in range(1, len( self.dict[word]["words"] )):
					seg.insert(index + i, self.dict[word]["words"][i])
				index += len( self.dict[word]["words"] ) - 1
			index += 1
		return seg

	# combine some single characters into words
	def combineSingle(self, seg, threshold = 8.0e-5):
		combine = ""
		index = 0
		# check each single character
		# cannot modify iterator if use for loop
		while index < len(seg):
			word = seg[index]
			# if a single character, and it is unknown or rare
			if 1 == len(word) and (not (word in self.dict) or self.dict(word) <= threshold ):
				# combine it
				combine += word
			# if not, combine previous ones
			elif "" != combine:
				# if a single character
				if 1 == len(combine):
					# get value of previous, next words
					previous = - sys.float_info.max
					if index <= 1:
						previous = sys.float_info.max
					elif seg[index - len(combine) - 1] in self.dict:
						previous = self.dict( seg[index - len(combine) - 1] )
					next_word = - sys.float_info.max
					if seg[index - len(combine) + 1] in self.dict:
						next_word = self.dict( seg[index - len(combine) + 1] )
					# if previous is lower, combine with it
					if previous <= threshold and previous < next_word:
						self.printTest("combine1: " + seg[index - len(combine) - 1] + combine + " begin: " + seg[index - len(combine) - 1])
						seg[index - len(combine) - 1] += combine
						del seg[index - len(combine) : index]
						index -= len(combine)
					# if next is lower, combine with it
					elif next_word <= threshold and previous > next_word:
						self.printTest("combine2: " + combine + seg[index - len(combine) + 1] + " begin: " + combine)
						seg[index - len(combine) + 1] = combine + seg[index - len(combine) + 1]
						del seg[index - len(combine) : index]
						index -= len(combine)
				# combine single ones
				else:
					self.printTest("combine: " + combine + " begin: " + seg[index - len(combine)])
					seg[index - 1] = combine
					del seg[index - len(combine) : index - 1]
					index -= len(combine)
				combine = ""
			index += 1
		return seg

	# put missing words back
	def restoreMissing(self, seg, sentence):
		for index in range( len(seg) ):
			missing = ""
			# find missing word
			for letter in seg[index]:
				if len(sentence) == 0:
					break
				if letter != sentence[0]:
					missing += sentence[0]
				sentence = sentence[1 :]
			# put missing words back
			if "" != missing:
				self.printTest("missing: " + missing)
				seg.insert(index, missing)
				index -= 1
			if len(sentence) == 0:
				break
		return seg

	# find wrong segmentation
	def compareResult(self):
		
		with open("output.log") as f:
		    output = list(f)
			
		with open("data/reference") as ref:
			reference = list(ref)
		
		if len(output) != len(reference):
		    raise ValueError("Error: output and reference do not have identical number of lines")
		else:
		    with open("compare.log","w") as compare:
				for index,sent in enumerate(reference):
					compare.write("output:\t" + output[index] + "\nrefer:\t" + sent + "\n\n")
					#compare.write("count: " + repr(count) + "\n")

s = Segmenter(opts.input, [opts.counts1w, opts.counts2w])
ans = s.run()
s.output_file.close()
s.test_file.close()
s.compareResult()


# output the segmented text
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
# with open(opts.input) as f:
    # for line in f:
        # utf8line = unicode(line.strip(), 'utf-8')
        # output = [i for i in utf8line]  # segmentation is one word per character in the input
        # #print " ".join(output)


sys.stdout = old









