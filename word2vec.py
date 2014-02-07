#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Added parts by Franziska Horn <cod3licious@gmail.com>

"""
Deep learning via word2vec's "hierarchical softmax skip-gram model" [1]_.

The training algorithm was originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

**Install Cython with `pip install cython` before to use optimized word2vec training** (70x speedup [2]_).

Initialize a model with e.g.::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the model. Some of them
are already built-in::

  >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.similarity('woman', 'man')
  0.73723527

  >>> model['computer']  # raw numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

and so on.

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

import logging
import sys
import os
import heapq
import time
import itertools
from math import sqrt as msqrt

from numpy import dot, random, dtype, get_include, float32 as REAL,\
    uint32, seterr, array, zeros, uint8, vstack, argsort, fromstring, sqrt, newaxis

logger = logging.getLogger("word2vec.word2vec")
import utils  # utility fnc for pickling, common scipy operations etc

MODEL = 'cbow' # 'skipgram' or 'cbow' or ...
TRAINING = 'negsam' # 'hsoftm' or 'negsam'

if MODEL == 'skipgram':
    if TRAINING == 'hsoftm':
        from trainmodel import train_bskipgramHSM as train_sentences
    elif TRAINING == 'negsam':
        from trainmodel import train_bskipgramNEG as train_sentences
    else:
        raise RuntimeError("TRAINING not known!")
elif MODEL == 'cbow':
    if TRAINING == 'hsoftm':
        from trainmodel import train_cbowHSM as train_sentences
    elif TRAINING == 'negsam':
        from trainmodel import train_cbowNEG as train_sentences
    else:
        raise RuntimeError("TRAINING not known!")
else:
    raise RuntimeError("MODEL not known!")


class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class Word2Vec(utils.SaveLoad):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `save_word2vec_format()` and `load_word2vec_format()`.

    """
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5, seed=1, min_alpha=0.0001):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (utf8 strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
        `seed` = for the random number generator.
        `min_count` = ignore all words with total frequency lower than this.

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.table = None # for negative sampling
        self.layer1_size = int(size)
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.min_count = min_count
        self.min_alpha = min_alpha
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)


    def make_table(self, table_size=100000000., power = 0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines. 
        Called internally from `build_vocab()`.

        """
        logger.info("constructing a table with noise distribution from %i words" % len(self.vocab))
        # table (= list of words) of noise distribution for negative sampling
        vocab_size = len(self.index2word)
        self.table = zeros(table_size)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.vocab[self.index2word[widx]].count**power / train_words_pow
        for tidx in range(int(table_size)):
            self.table[tidx] = widx
            if tidx/table_size > d1:
                widx += 1
                d1 += self.vocab[self.index2word[widx]].count**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1


    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words" % len(self.vocab))

        # build the huffman tree
        heap = self.vocab.values()
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i" % max_depth)

    def build_vocab(self, sentences, threshold=0):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of utf8 strings.

        """
        logger.info("collecting all words and their counts")
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i unique words" %
                    (sentence_no, total_words, len(vocab)))
            for word in sentence:
                total_words += 1
                try:
                    vocab[word].count += 1
                except KeyError:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i unique words from a corpus of %i words and %i sentences" %
            (len(vocab), total_words, sentence_no + 1))

        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, v in vocab.iteritems():
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.index2word.append(word)
                self.vocab[word] = v
        logger.info("total of %i unique words after removing those with count < %s" % (len(self.vocab), self.min_count))

        # add probabilities for subsampling (if threshold > 0)
        if threshold > 0:
            total_words = float(sum(v.count for v in self.vocab.itervalues()))
            for word in self.vocab:
                # formula from paper
                #self.vocab[word].prob = max(0.,1.-msqrt(threshold*total_words/self.vocab[word].count))
                # formula from code
                self.vocab[word].prob = (msqrt(self.vocab[word].count / (threshold * total_words)) + 1.) * (threshold * total_words) / self.vocab[word].count
        else:
            # if prob is 0, word wont get discarded 
            for word in self.vocab:
                self.vocab[word].prob = 0.
        # add info about each word's Huffman encoding
        if TRAINING == 'hsoftm':
            self.create_binary_tree()
        # build the table for drawing random words (for negative sampling)
        else:
            self.make_table()
        self.reset_weights()


    def train(self, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of utf8 strings.

        """
        logger.info("training model on %i vocabulary and %i features" % (len(self.vocab), self.layer1_size))
        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), 1.0
        if not total_words:
            total_words = sum(v.count for v in self.vocab.itervalues())
        # convert input string lists to Vocab objects (or None for OOV words)
        no_oov = ([self.vocab.get(word, None) for word in sentence] for sentence in sentences)
        # run in chunks of e.g. 100 sentences (= 1 job) 
        for job in utils.grouper(no_oov, chunksize):
            # update the learning rate before every job
            alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count / total_words))
            # how many words did we train on? out-of-vocabulary (unknown) words do not count
            job_words = sum(train_sentences(self, sentence, alpha) for sentence in job)
            word_count += job_words
            # report progress
            elapsed = time.time() - start
            if elapsed >= next_report:
                logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                    (100.0 * word_count / total_words, alpha, word_count / elapsed if elapsed else 0.0))
                next_report = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports
        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count, elapsed, word_count / elapsed if elapsed else 0.0))
        return word_count


    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        random.seed(self.seed)
        self.syn0 = utils.zeros_aligned((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn1 = utils.zeros_aligned((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0 += (random.rand(len(self.vocab), self.layer1_size) - 0.5) / self.layer1_size
        self.syn0norm = None

    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, basestring) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, basestring) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, basestring) else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if word in self.vocab:
                mean.append(weight * utils.unitvec(self.syn0[self.vocab[word].index]))
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = utils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]


    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Example::

          >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
          'cereal'

        """
        words = [word for word in words if word in self.vocab]  # filter out OOV words
        logger.debug("using words %s" % words)
        if not words:
            raise ValueError("cannot select a word from an empty list")
        # which word vector representation is furthest away from the mean?
        vectors = vstack(utils.unitvec(self.syn0[self.vocab[word].index]) for word in words).astype(REAL)
        mean = utils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, words))[0][1]


    def __getitem__(self, word):
        """
        Return a word's representations in vector space, as a 1D numpy array.

        Example::

          >>> trained_model['woman']
          array([ -1.40128313e-02, ...]

        """
        return self.syn0[self.vocab[word].index]


    def __contains__(self, word):
        return word in self.vocab


    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Example::

          >>> trained_model.similarity('woman', 'man')
          0.73723527

          >>> trained_model.similarity('woman', 'woman')
          1.0

        """
        return dot(utils.unitvec(self[w1]), utils.unitvec(self[w2]))


    def init_sims(self):
        if getattr(self, 'syn0norm', None) is None:
            logger.info("precomputing L2-norms of word weight vectors")
            self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)


    def accuracy(self, questions, restrict_vocab=30000):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30,000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = dict(sorted(self.vocab.iteritems(), key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in ok_vocab.itervalues())

        def log_accuracy(section):
            correct, incorrect = section['correct'], section['incorrect']
            if correct + incorrect > 0:
                logger.info("%s: %.1f%% (%i/%i)" %
                    (section['section'], 100.0 * correct / (correct + incorrect),
                    correct, correct + incorrect))

        sections, section = [], None
        for line_no, line in enumerate(open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = [word.lower() for word in line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line))
                    continue

                ignore = set(self.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                for index in argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                        break
                section['correct' if predicted == expected else 'incorrect'] += 1
        if section:
            # store the last section, too
            sections.append(section)
            log_accuracy(section)

        total = {'section': 'total', 'correct': sum(s['correct'] for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
        log_accuracy(total)
        sections.append(total)
        return sections


    def __str__(self):
        return "Word2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)



class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in open(fname):
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', 1000
        with open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split()) # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(' ')  # the last token may have been split in two... keep it for the next iteration
                words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]


class LineSentence(object):
    def __init__(self, source):
        """Simple format: one sentence = one line; words already preprocessed and separated by whitespace.

        source can be either a string or a file object

        Thus, one can use this for just plain files:

            sentences = LineSentence('myfile.txt')

        Or for compressed files:

            sentences = LineSentence(bz2.BZ2File('compressed_text.bz2'))
        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                yield line.split()
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            for line in open(self.source):
                yield line.split()



# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    infile = sys.argv[1]
    from gensim.models.word2vec import Word2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    # model = Word2Vec(LineSentence(infile), size=200, min_count=5, workers=4)
    model = Word2Vec(Text8Corpus(infile), size=200, min_count=5, workers=1)

    if len(sys.argv) > 3:
        outfile = sys.argv[3]
        model.save(outfile + '.model')
        model.save_word2vec_format(outfile + '.model.bin', binary=True)
        model.save_word2vec_format(outfile + '.model.txt', binary=False)

    if len(sys.argv) > 2:
        questions_file = sys.argv[2]
        model.accuracy(sys.argv[2])

    logging.info("finished running %s" % program)
