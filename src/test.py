import datetime

def reverse_vocab(vocab):
  reversed_vocab = {}
  for token, idx in vocab.items():
    reversed_vocab[idx] = token
  return reversed_vocab

def idx_to_token(token, reversed_vocab):
  return reversed_vocab.get(token)



start_datetime = datetime.datetime.now()
vocabulary = {'AS': 410, 'AT': 375, 'AW': 75, 'AV': 441, 'AY': 457, 'AX': 170, 'AZ': 462, 'VK': 103}
reversed_vocab = reverse_vocab(vocabulary)
print(vocabulary)
print(reversed_vocab)
print(idx_to_token(410, reversed_vocab))


print(datetime.datetime.now()-start_datetime)