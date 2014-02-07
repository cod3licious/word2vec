README

This is an extension / modification of the gensim word2vec python port (see here: http://radimrehurek.com/gensim/ and here: http://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim/).

This code is still under construction, it comes as is, with absolutely no warranty, etc.
I'm not quite sure about licenses and stuff; the original code by Radim is licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html, for my parts, please don't use them for military, NSA, and related purposes.

While this code is a derivative of the gensim word2vec code, it is actually detached from it (but it should be pretty easy to integrate the relevant parts back in). It consists of 3 parts
- utils.py simply includes the utils and matutils functions needed from gensim.
- trainmodel.py contains 6 functions in the style of the original python train_sentence function and one of them is imported as such in word2vec depending on the settings. One of the functions is basically the original function, renamed to train_skipgramHSM, then there is train_skipgramNEG, which implements negative sampling, then there are 2 more skipgram functions (starting with a b), again with HSM and NEG, however they operate in batch mode, by training on all the words in the word's window at once (this is around 3 times faster, however the accuracy is a little lower (25.2% instead of 27.5% for HSM, for NEG it's 17.9% for batch and 15.9% otherwise though)). And then there are the two cbow functions. All implementations are close to the original C code (as far as I could understand it without any comments...;-)). 
- word2vec.py is pretty much the same, however I've removed the threading (but it should be pretty easy to add that back in) and I've added some probabilities at the end of build_vocab, used for subsampling if threshold > 0, and I've added a function make_table, which makes a table with word indexes similar as in the C code, used for the negative sampling.

The code seems to work fine, i.e. it achieves similar accuracies as the original word2vec C implementation (cbow HSM: 14.7% (original was 15.59%), cbow NEG: 16.2% (original was 16.32), skipgram NEG: 17.9 (original was 15.64)), however it would be really great if a second pair of eyes could look over it.

Feedback is very much appreciated! [gmail: cod3licous]
