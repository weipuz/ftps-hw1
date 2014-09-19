
import sys, codecs, optparse, os, math, heapq

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
        #else: return self.missingfn(key, self.N)             #pw=1/N when no character find
        elif len(key) == 1: return self.missingfn(key, self.N)  #pw=1/N when single character is missing 
        else: return None                                       #pw=None when a character longer than 1 is missing 

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)         #use counts1w as the total Pw function


class Segmenter():
	"segmenter"

	# load a input file
	def __init__(self, file):
		self.lines = [ unicode(line.strip(), 'utf-8') for line in open(file) ]
		
		
	#def __cmp__(self, other):    #commands to change the compare key in heapq . By default heapq will compare the first number in a list
    #return cmp(self.intAttribute, other.intAttribute)

	# print the original input file
	def printText(self):
		for line in self.lines:
			print line

	# segment the entire input file
	def run(self):
		for line in self.lines:
			print self.segmentLine(line)
			

	# segment a sentence
	def segmentLine(self, line):
		if 0 == len(line):
			return ""
		chart = [None] * (len(line) )   #chart to restore every entry containing the information of every newword 
		heap = []                       #initialize heap queue
		
		####### for each word that matches input at position 0 #########
		for i in range(1, len(line)+1 ):
			word = line[: i]
			
			###### insert Entry(word, 0, logPw(word), None) into heap  #########
			if Pw(word) != None:
			    entry = (word, 0, math.log10(Pw(word)), None)
			    heapq.heappush(heap, entry)
		# while heap is nonempty:
		while len(heap):
			#### entry = top entry in the heap
			#### sorting algorithm need to be defined here 
			
			from operator import itemgetter 
			heap.sort(key=itemgetter(2), reverse=True)
			
			
			entry = heapq.heappop(heap)
			#print entry
			# get the endindex based on the length of the word in entry
			end_index = len(entry[0]) + entry[1]-1
			#print "end_index", end_index
			# if chart[endindex] has a previous entry, preventry
			if end_index < len(chart)-1 and None != chart[end_index] and len(chart[end_index]) == 4 and None != chart[end_index][3]:
				prev_entry = chart[end_index]
				# if entry has a higher probability than preventry:
				if entry[2] > prev_entry[2]:
					# chart[endindex] = entry
					chart[end_index] = entry  
				else:
				      continue  
				
			else:
				chart[end_index] = entry
			# for each newword that matches input starting at position endindex+1
			for i in range(end_index + 1, len(line)+1 ):
				new_word = line[end_index+1 : i]
				if Pw(new_word) != None:			
			      
				       # newentry = Entry(newword, endindex+1, entry.log-probability + logPw(newword), entry)
				       new_entry = (new_word, end_index+1, entry[2]+math.log10(Pw(new_word)), end_index)
				
				      # print new_word, math.log10(Pw(new_word)), new_entry
					   
				       # if newentry does not exist in heap:
				       if not(new_entry in heap):
					       # insert newentry into heap
					       heapq.heappush(heap, new_entry)
					       ###print new_entry[0], new_entry[1], new_entry[2]
		# finalindex is the length of input
		final_index = len(line)-1
		# finalentry = chart[finalindex]
		final_entry = chart[final_index]
		#print chart
		seg = []
		# Print out the argmax output by following the backpointer from finalentry until you reach the first word
		while None != final_entry[3]:
			seg.append(final_entry[0])
			
			final_entry = chart[final_entry[3]]
		seg.append(final_entry[0])
		return " ".join(seg[: : -1])



s = Segmenter(opts.input)
#print s.lines[2]
print s.segmentLine(s.lines[0])    #test command for segment the first line
                   #test command for segment the entire file


old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
# from io import open

#s.run() 
 
sys.stdout = old
