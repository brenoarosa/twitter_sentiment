from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/wikipedia/options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/wikipedia/elmo_pt_weights.hdf5"

#elmo = Elmo(options_file, weight_file, num_output_representations=4, dropout=0.1)
elmo = Elmo(options_file, weight_file, num_output_representations=4)

# use batch_to_ids to convert sentences to character ids
# sentences = [['First', 'sentence', '.'], ['Another', 'sentence','is','this','.'],['A', 'good','sentence','.']]
sentences = [["Eu", "comi", "uma", "maçã", "no", "café"],["Eu", "comi", "uma", "laranja", "no", "jantar"]]
#Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters (len(batch), max sentence length, max word length).
character_ids = batch_to_ids(sentences)

print (character_ids,character_ids.size())
embeddings = elmo(character_ids)

emb=embeddings['elmo_representations']
print (type(emb))
print (len(emb))

# the lower layer 0;  for the word  'ate'
s1 = emb[0][0][1]
s2 = emb[0][1][1]
print (s1, s1.size())

import scipy
scipy.spatial.distance.cosine(s1.detach().numpy(),s2.detach().numpy())

s1 = emb[3][0][1]
s2 = emb[3][1][1]
scipy.spatial.distance.cosine(s1.detach().numpy(),s2.detach().numpy())
