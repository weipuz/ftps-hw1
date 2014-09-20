
import sys, codecs, optparse, os, math
# parse the input command lines
optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."


    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.additional_dict = [u"sdfsdfd"]
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
        elif key in self.additional_dict:
			return -1.0
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)

# output the segmented text
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        output = [i for i in utf8line]  # segmentation is one word per character in the input
        #print " ".join(output)
sys.stdout = old

class Segmenter():
	"segmenter"

	# load a input file
	def __init__(self, file):
		self.text = [ unicode(text.strip(), "utf-8") for text in open(file) ]
		self.test_file = codecs.open("test.log", "w", "utf-8")
		self.output_file = codecs.open("output.log", "w", "utf-8")

	# destructor
	def __exit__(self, type, value, traceback):
		self.test_file.close()

	# print the test comments
	def printTest(self, output):
		self.test_file.write(output + "\n")

	# print output to file
	def printOutput(self, output):
		self.output_file.write(output + "\n")

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
		for i in range(1, self.min(len(sentence) + 1, Pw.maxlen)):
			word = sentence[: i]
			p = Pw(word)
			if p != None:
				# insert Entry(word, 0, logPw(word), None) into heap
				entry = {"word": word, "start": 0, "logprob": math.log10(p), "back": None} #(0, math.log10(p), word, None)
				self.printTest("Adding: " + entry["word"] + " " + repr(entry["logprob"]) )
				heap.append(entry)

		## Iteratively fill in chart[i] for all i ##

		# while heap is nonempty:
		while len(heap):
			# entry = top entry in the heap
			entry = self.heapPop(heap)
			if None == entry:
				break
			self.printTest( "pop: word=" + entry["word"] + " logprob=" + repr(entry["logprob"]) )
			# get the endindex based on the length of the word in entry
			end_index = entry["start"] + len(entry["word"]) - 1
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
				p = Pw(new_word)
				if None != p:
					# newentry = Entry(newword, endindex+1, entry.log-probability + logPw(newword), entry)
					new_entry = {"word": new_word, "start": end_index + 1, "logprob": entry["logprob"] + math.log10(p), "back": entry} #(end_index + 1, entry[1] + math.log10(p), new_word, entry)
					# if newentry does not exist in heap:
					if not(new_entry in heap):
						self.printTest("endIndex= " + repr(end_index) + " : newEntry= " + self.strEntry(new_entry))
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

		# check each single character
		combine = ""
		for index, word in enumerate(seg):
			# if a unknown single character
			if 1 == len(word) and not (word in Pw):
				combine += word
			# if not, combine previous ones
			elif "" != combine:
				# if a single character, combine with previous word
				if 1 == len(combine) and index > 1 and len(seg[index - len(combine) - 1]) == 1:
					self.printTest("combine-1: " + seg[index - len(combine) - 1] + combine + " begin: " + seg[index - len(combine) - 1])
					seg[index - len(combine) - 1] += combine
					del seg[index - len(combine) : index]
					index -= len(combine)
				# combine single ones
				else:
					self.printTest("combine: " + combine + " begin: " + seg[index - len(combine)])
					seg[index - 1] = combine
					del seg[index - len(combine) : index - 1]
					index -= len(combine)
				combine = ""

		# put missing words back
		#for index in range( len(seg) ):
			#missing = ""
			## find missing word
			#for letter in seg[index]:
				#if len(sentence) == 0:
					#break
				#if letter != sentence[0]:
					#missing += sentence[0]
				#sentence = sentence[1 :]
			#if "" != missing:
				#self.printTest("missing: " + missing)
				#seg.insert(index, missing)
				#-- index
			#if len(sentence) == 0:
				#break

		ans = " ".join(seg)
		self.printTest(ans + "\n")
		return ans

	# find wrong segmentation
	def compareResult(self):
		output = [ unicode(text.strip(), "utf-8") for text in open("output.log") ]
		reference = [ unicode(text.strip(), "utf-8") for text in open("data/reference") ]
		compare = codecs.open("compare.log", "w", "utf-8")
		for index, sent in enumerate(reference):
			if sent != output[index]:
				compare.write("output:\t" + output[index] + "\nrefer:\t" + sent + "\n\n")

s = Segmenter(opts.input)
#print s.segmentSent(s.text[4])
ans = s.run()
s.compareResult()















