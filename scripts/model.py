import tensorflow as tf
tf.enable_eager_execution()


import numpy as np
import os
import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   *TRAINING TEXT SETUP*   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #


# download shakespeare.txt
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file).read()

# Creating a mapping from unique characters to indices
vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100

# Create training examples / targets
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = chunks.map(split_input_target)

# for input_example, target_example in  dataset.take(1):
#   print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#   print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print("Step {:4d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    
# Batch size 
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
#       *MODEL SETUP*       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(Model, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      self.gru = tf.keras.layers.CuDNNGRU(self.units, 
                                          return_sequences=True, 
                                          recurrent_initializer='glorot_uniform',
                                          stateful=True)
    else:
      self.gru = tf.keras.layers.GRU(self.units, 
                                     return_sequences=True, 
                                     recurrent_activation='sigmoid', 
                                     recurrent_initializer='glorot_uniform', 
                                     stateful=True)

    self.fc = tf.keras.layers.Dense(vocab_size)
        
  def call(self, x):
    embedding = self.embedding(x)
    
    # output at every time step
    # output shape == (batch_size, seq_length, hidden_size) 
    output = self.gru(embedding)
    
    # The dense layer will output predictions for every time_steps(seq_length)
    # output shape after the dense layer == (seq_length * batch_size, vocab_size)
    prediction = self.fc(output)
    
    # states will be used to pass at every step to the model while training
    return prediction
  
  
# **Instantiate the Model** #

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
units = 1024

model = Model(vocab_size, embedding_dim, units)

# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()

# Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)
  
  
# **Setup Training Checkpoints**

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# Checkpoint instance
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


## TRAIN MODEL ##

# Build and Summerize Model
model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
model.summary()

# Training Loop

# Training step
EPOCHS = 1

for epoch in range(EPOCHS):
    start = time.time()
    
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    
    for (batch, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions = model(inp)
              loss = loss_function(target, predictions)
              
          grads = tape.gradient(loss, model.variables)
          optimizer.apply_gradients(zip(grads, model.variables))

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                            batch,
                                                            loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))