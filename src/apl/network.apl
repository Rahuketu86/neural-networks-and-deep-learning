⍝ ---------------------------------------
⍝ Stochastic gradient descent learning
⍝ ---------------------------------------

⍝ The sigmoid function
sigmoid ← { ÷1+*-⍵ }

⍝ Derivative of sigmoid
sigmoid_prime ← { s × 1 - (s ← sigmoid ⍵) }

⍝ Generate a random permutation.
⍝ [random_perm n] generates a random permutation of ⍳n.
random_perm ← { ⍋{?10×a}¨⍳(a←⍵) }

⍝ Initialise a network layer.
⍝ [prev_sz network_layer sz] returns an initialized network layer
⍝ (B,W) of biases and weights, where B is a vector of size sz and W is a
⍝ matrix of dimension prev_sz × sz.
network_layer ← {
  sz ← ⍵
  prev_sz ← ⍺
  biases ← sz ⍴ 0.5
  weights ← sz prev_sz ⍴ 0.5
  biases weights
}

⍝ Initialise a network given a configuration (a vector of neuron
⍝ numbers, one number for each layer).

network3 ← {
  sz1 ← ⍵[1] ⋄ sz2 ← ⍵[2] ⋄ sz3 ← ⍵[3]
  layer2 ← sz1 network_layer sz2
  layer3 ← sz2 network_layer sz3
  layer2 layer3
}

feedforward_layer ← {
  b ← ⍺[1] ⋄ w ← ⍺[2]
  sigmoid b + w +.× ⍵
}

⍝ [(B W) feedforward3 a] returns the output of the network (B,W) given
⍝ the input a.
feedforward3 ← {
  layers ← ⍺
  layer2 ← layers[1]
  layer3 ← layers[2]
  a ← ⍵
  a ← layer2 feedforward_layer a
  a ← layer3 feedforward_layer a
  a
}

⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝
⍝ Stochastic Gradient Descent
⍝
⍝ [(network epochs mini_batch_size eta) SGD training_data] returns a
⍝ new network trained with the provided training data. The input `epochs`
⍝ specifies the number of random shuffes of the training data. The input
⍝ `mini_batch_size` specifies the training data split size in each epoch. The
⍝ input `eta` is the training rate.
⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝

SGD ← {
  training_input ← ⍵[1]
  training_output ← ⍵[2]
  network ← ⍺[1]
  epochs ← ⍺[2]
  mini_batch_size ← ⍺[3]
  eta ← ⍺[4]
  n ← ≢ training_input

  loop ← {
    ⎕ ← 'ping'
    perm ← random_perm n
    training_input ← training_input[perm]
    training_output ← training_output[perm]
    network ← ⍵
    network
  }
  (loop⍣epochs) network
}

test1 ← {
  ⍵
  ⎕ ← sigmoid 5
  ⎕ ← sigmoid 0
  ⎕ ← sigmoid_prime 5
  ⎕ ← sigmoid_prime 0

  n ← network3 4 200 10
  a ← 3 2 3 1
  a ← n feedforward3 a
  r ← +/ a
}

test2 ← {
  epochs ← ⍵
  mini_batch_size ← 10
  eta ← 3.0
  training_input ← ⍳100
  training_output ← 10+⍳100
  training_data ← training_input training_output
  n ← network3 4 200 10
  n ← (n epochs mini_batch_size eta) SGD training_data
  0
}

test2 10

⍝ test1 0
