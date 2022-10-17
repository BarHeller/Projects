import math
from collections import defaultdict
import numpy as np
import random
import argparse

grammar_raw = """# Symbols in the grammar are case-sensitive.
#
# This grammar uses a convention that
#    - terminals are lowercase          (president)
#    - preterminals are capitalized     (Noun)
#    - other nonterminals are all-caps  (NP)
#
# This convention just makes grammars more readable to humans.  Thus:
#
#    - When *you* are writing grammars, you should
#      follow this convention unless you have a good reason not to.
#
#    - But the  *program* should still work with grammars that don't
#      follow this convention.  So how can the program reliably tell
#      the difference between terminal and nonterminal symbols?  If
#      there is at least one rule for rewriting a symbol, then that
#      symbol is a nonterminal and should be rewritten.
#######################

# Rules for creating full sentences.



# The basic grammar rules.  Here's what the abbreviations stand for:
#    S  = sentence
#    NP = noun phrase
#    VP = verb phrase
#    PP = prepositional phrase
#    Det = determiner (sometimes called "article")
#    Prep = preposition
#    Adj = adjective

1	S	NP VP
1	VP	Verb NP
1	NP	Det Noun
1	NP	NP PP
1	PP	Prep NP
1	Noun	Adj Noun

# Vocabulary.  Your program can see that "ate" is a terminal
# symbol because there exists no rule for rewriting it.
# Any symbol that can rewrite as a terminal (or a string of
# terminals, like "chief of staff") is called a "preterminal."  Notice
# that a preterminal is a special kind of nonterminal.

1	Verb	ate
1	Verb	wanted
1	Verb	kissed
1	Verb	understood
1	Verb	pickled
1	Verb	worked

1	Det	the
1	Det	a
1	Det	every

1	Noun	president
1	Noun	airline
1	Noun	sandwich
1	Noun	pickle
1	Noun	floor

1	Adj	fine
1	Adj	delicious
1	Adj	perplexed
1	Adj	sharper
1	Adj	pickled

1	Prep	with
1	Prep	on
1	Prep	under
1	Prep	in
""" 

class PCFG(object):
	def __init__(self):
		self._rules = defaultdict(list)
		self._sums = defaultdict(float)

	def add_rule(self, lhs, rhs, weight):
		assert(isinstance(lhs, str))
		assert(isinstance(rhs, list))
		self._rules[lhs].append((rhs, weight))
		self._sums[lhs] += weight

	@classmethod
	def from_file_assert_cnf(cls):
		grammar = PCFG()
		for line in grammar_raw.split("\n"):
			line = line.split("#")[0].strip()
			if not line: continue
			w,l,r = line.split(None, 2)
			r = r.split()
			w = float(w)
			if len(r) > 2:
				raise Exception("Grammar is not CNF, right-hand-side is: " + str(r))
			if len(r) <= 0:
				raise Exception("Grammar is not CNF, right-hand-side is empty: " + str(r))
			grammar.add_rule(l,r,w)
		for lhs, rhs_and_weights in grammar._rules.items():
			for rhs, weight in rhs_and_weights:
				if len(rhs) == 1 and not grammar.is_terminal(rhs[0]):
					raise Exception("Grammar has unary rule: " + str(rhs))
				elif len(rhs) == 2 and (grammar.is_terminal(rhs[0]) or grammar.is_terminal(rhs[1])):
					raise Exception("Grammar has binary rule with terminals: " + str(rhs))

		return grammar

	def is_terminal(self, symbol): return symbol not in self._rules

	def is_preterminal(self, rhs):
		return len(rhs) == 1 and self.is_terminal(rhs[0])

	def gen(self, symbol):
		if self.is_terminal(symbol): return symbol
		else:
			expansion = self.random_expansion(symbol)
			return " ".join(self.gen(s) for s in expansion)

	def gentree(self, symbol):
		"""
			Generates a derivation tree from a given symbol
		"""
		if self.is_terminal(symbol): return symbol
		if self.is_preterminal(symbol):
			return "("+symbol+" "+self.gentree(self.random_expansion(symbol)) +")"
		else:
			expansion = self.random_expansion(symbol)
			return "("+symbol+" "+" ".join(self.gentree(s) for s in expansion)+")"
		return ""

	def random_sent(self):
		return self.gen("S")

	def random_tree(self):
		return self.gentree("S")

	def random_expansion(self, symbol):
		"""
		Generates a random RHS for symbol, in proportion to the weights.
		"""
		p = random.random() * self._sums[symbol]
		for r,w in self._rules[symbol]:
			p = p - w
			if p < 0: return r
		return r


def cky(pcfg, sent):
	
	pi = defaultdict(float)
	bp = defaultdict(list)
	sent = sent.split(" ")
	### YOUR CODE HERE
	raise NotImplementedError 
	### END YOUR CODE
	return bp,pi
	
	
def gen_result(bp,i,j,TAG):
	s, tup = bp[i, j, TAG]
	if len(tup) > 1:
		Y, Z = tup
		return f"({TAG} {gen_result(bp,i,s,Y)} {gen_result(bp,s+1,j,Z)})"
	else:
		return f"({TAG} {tup[0]})"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=['gen', 'inference'])
	parser.add_argument("--sent", type=str,default="the president ate the delicious sandwich")

	args = parser.parse_args()
	pcfg = PCFG.from_file_assert_cnf()
	if args.mode=="gen":
		print(pcfg.random_sent())
	else:
		bp,pi = cky(pcfg, args.sent)
		if(pi[0,len(args.sent.split(" "))-1,'S']>0):
			print(gen_result(bp,0,len(args.sent.split(" "))-1,'S'))
		else:
			print("FAILED TO PARSE!")

