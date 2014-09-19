
import sys, codecs, optparse, os, math, heapq
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
		self.lines = [ unicode(line.strip(), 'utf-8') for line in open(file) ]
		self.test_file = codecs.open('test.log', 'w', "utf-8")
		self.output_file = codecs.open('output.log', 'w', "utf-8")

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

	def strEntry(self, entry):
		s = "chartEntry(" + entry[2] + ", start=" + repr(entry[0]) + ", end=, logprob=" + repr(entry[1]) + ", backptr="
		if None != entry[3]:
			s += entry[3][2]
		else:
			s += "None"
		return s

	# segment the entire input file
	def run(self):
		ans = []
		for line in self.lines:
			ans0 = self.segmentLine(line)
			self.printOutput(ans0)
			ans.append(ans0)
		return ans

	# segment a sentence
	def segmentLine(self, line):
		# line(input): the input sequence of characters
		if 0 == len(line):
			return ""
		self.printTest("input: " + line)
		# chart: the dynamic programming table to store the argmax for every prefix of input, indexed by character position in input
		chart = [None] * (len(line) + 1)
		# heap: a list or priority queue containing the entries to be expanded, sorted on start-position or log-probability
		heap = []
		# entry: each entry in the chart has four components: Entry(word, start-position, log-probability, back-pointer), the back-pointer in each entry links it to a previous entry that it extends
		# change to Entry(start-position, log-probability, word, back-pointer)

		## Initialize the heap ##

		# for each word that matches input at position 0
		for i in range(1, self.min(len(line) + 1, Pw.maxlen)):
			word = line[: i]
			p = Pw(word)
			if p != None:
				# insert Entry(word, 0, logPw(word), None) into heap
				entry = (0, math.log10(p), word, None)
				self.printTest("Adding: " + entry[2] + " " + repr(entry[1]) )
				heapq.heappush(heap, entry)

		## Iteratively fill in chart[i] for all i ##

		# while heap is nonempty:
		while len(heap):
			# entry = top entry in the heap
			entry = heapq.heappop(heap)
			self.printTest( "pop: word=" + entry[2] + " logprob=" + repr(entry[1]) )
			# get the endindex based on the length of the word in entry
			end_index = entry[0] + len(entry[2]) - 1
			# if chart[endindex] has a previous entry, preventry
			if end_index < len(chart) and None != chart[end_index] and len(chart[end_index]) == 4 and None != chart[end_index][3]:
				prev_entry = chart[end_index][3]
				# if entry has a higher probability than preventry:
				if entry[1] > prev_entry[1]:
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
			for i in range(end_index + 2, len(line) + 1):
				new_word = line[end_index + 1 : i]
				p = Pw(new_word)
				if None != p:
					# newentry = Entry(newword, endindex+1, entry.log-probability + logPw(newword), entry)
					new_entry = (end_index + 1, entry[1] + math.log10(p), new_word, entry)
					# if newentry does not exist in heap:
					if not(new_entry in heap):
						self.printTest("endIndex= " + repr(end_index) + " : newEntry= " + self.strEntry(new_entry))
						# insert newentry into heap
						heapq.heappush(heap, new_entry)

		## Get the best segmentation ##

		# finalindex is the length of input
		final_index = len(line) - 1
		# finalentry = chart[finalindex]
		final_entry = chart[final_index]
		seg = []
		# The best segmentation starts from finalentry and follows the back-pointer recursively until the first word
		while None != final_entry:
			self.printTest("final[ 0 ]: " + self.strEntry(final_entry))
			seg.append(final_entry[2])
			final_entry = final_entry[3]
		self.printTest("")
		return " ".join(seg[: : -1])


s = Segmenter(opts.input)
#print s.segmentLine(s.lines[4])
ans = s.run()
for item in ans:
	print item















