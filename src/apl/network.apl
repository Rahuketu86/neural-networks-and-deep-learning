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

cost_derivative ← {
  output_activations ← ⍺
  y ← ⍵
  output_activations - y
}

⍝ [zero_layer (B W)] returns a layer with zero's.
zero_layer ← {
  B ← { x ← ⍵ ⋄ 0 }¨⍵[1]
  W ← { x ← ⍵ ⋄ 0 }¨⍵[2]
  B W
}

⍝ update_mini_batch ← {
⍝   network ← ⍺            ⍝ array of biases and weights
⍝   layer2 ← network[1]
⍝   layer3 ← network[2]

⍝   backprop3 ← {
⍝     x ← ⍺   ⍝ vector
⍝     y ← ⍵   ⍝ vector; different length
⍝     nabla2 ← zero_layer layer2
⍝     nabla3 ← zero_layer layer3
⍝     (x-x) (y-y)
⍝   }

⍝   mini_batch_xs ← ⍵[1]   ⍝ array of xs
⍝   mini_batch_ys ← ⍵[2]   ⍝ array of ys
⍝   eta ← ⍵[3]             ⍝ learning rate

⍝   nabla2 ← zero_layer layer2
⍝   nabla3 ← zero_layer layer3
⍝   nabla_zero ← zero_network network
⍝   delta_nablas ← mini_batch_xs backprop3 mini_batch_ys
⍝   nabla ←
⍝ }

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
  x ← ⍵
  ⎕ ← sigmoid 5
  ⎕ ← sigmoid 0
  ⎕ ← sigmoid_prime 5
  ⎕ ← sigmoid_prime 0

  n ← network3 4 200 10
  a ← 3 2 3 1
  a ← n feedforward3 a
  ⎕ ← a
  r ← +/ a
  r
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

test1 0

⍝ test1 0
