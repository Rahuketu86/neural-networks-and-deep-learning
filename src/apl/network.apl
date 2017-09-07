⍝ ---------------------------------------
⍝ Stochastic gradient descent learning
⍝ ---------------------------------------

⍝ The sigmoid function
sigmoid ← { ÷1+*-⍵ }

⍝ Derivative of sigmoid
sigmoid_prime ← { s × 1 - (s ← sigmoid ⍵) }

⍝ Generate a random permutation.
⍝ [random_perm n] generates a random permutation of ⍳n.
⍝ random_perm ← { ⍋{?10×a}¨⍳(a←⍵) }

random_perm ← {
  i ← ⍵[1]
  n ← ⍵[2]
  xs ← ⍳n
  i ⌽ i ↓ xs, i ↑ xs
}

⍝ Initialise a network layer.
⍝ [prev_sz layer_new sz] returns an initialized network layer
⍝ (B,W) of biases and weights, where B is a vector of size sz and W is a
⍝ matrix of dimension prev_sz × sz.
layer_new ← {
  sz ← ⍵
  prev_sz ← ⍺
  biases ← sz ⍴ 0.5
  weights ← sz prev_sz ⍴ 0.5
  biases weights
}

⍝ [zero_layer (B W)] returns a layer with zero's.
zero_layer ← {
  B ← { x ← ⍵ ⋄ 0.0 }¨⍵[1]
  W ← { x ← ⍵ ⋄ 0.0 }¨⍵[2]
  B W
}

⍝ [layer_sum (biasesVector weightsVector)] returns a layer computed by
⍝ horizontally reducing the biasesVector and the weightsVector.
layer_sum ← {
  biasesVector ← ⍵[1]
  weightsVector ← ⍵[2]
  (+⌿biasesVector) (⍉+/⍉weightsVector)
}

layer_feedforward ← {
  b ← ⍺[1] ⋄ w ← ⍺[2]
  sigmoid b + w +.× ⍵
}

⍝ Initialise a network given a configuration (a vector of neuron
⍝ numbers, one number for each layer).
network3_new ← {
  sz1 ← ⍵[1] ⋄ sz2 ← ⍵[2] ⋄ sz3 ← ⍵[3]
  layer2 ← sz1 layer_new sz2
  layer3 ← sz2 layer_new sz3
  layer2 layer3
}

network_sum ← {
  layer2 ← layer_sum ⍵[1]
  layer3 ← layer_sum ⍵[2]
  layer2 layer3
}

subM ← {
  (⍴⍺)⍴((,⍺)-,⍵)
}

network_sub ← {
  factor ← ⍵[1]
  network ← ⍵[2]
  nabla ← ⍵[3]
  layer2 ← network[1]
  layer3 ← network[2]
  nabla2 ← nabla[1]
  nabla3 ← nabla[2]
  b2 ← layer2[1] - factor × nabla2[1]
  w2 ← layer2[2] subM {factor × ⍵}¨nabla2[2]
  b3 ← layer3[1] - factor × nabla3[1]
  w3 ← layer3[2] subM {factor × ⍵}¨nabla3[2]
⍝   ⎕ ← 'Shape of layer2 weight:', (⍕ (⊃ ⍴⍴ layer2[2])), ' : ' , (⍕ (⍴ layer2[2])[1]) , ' x ' , (⍕ (⍴ layer2[2])[2])
⍝   ⎕ ← 'Shape of nabla2 weight:', (⍕ (⊃ ⍴⍴ nabla2[2])), ' : ' , (⍕ (⍴ nabla2[2])[1]) , ' x ' , (⍕ (⍴ nabla2[2])[2])
  (b2 w2) (b3 w3)
}

⍝ [(B W) feedforward3 a] returns the output of the network (B,W) given
⍝ the input a.
feedforward3 ← {
  layers ← ⍺
  layer2 ← layers[1]
  layer3 ← layers[2]
  a ← ⍵
  a ← layer2 layer_feedforward a
  a ← layer3 layer_feedforward a
  a
}

cost_derivative ← {
  output_activations ← ⍵[1]
  y ← ⍵[2]
  output_activations - y
}

vec2 ← {
  sh1 ← ⊃ ⍴ ⍵
  1 sh1 ⍴ ⍵
}

vec ← {
  sh1 ← (⍴ ⍵)[1]
  sh1 1 ⍴ ⍵
}

cons3 ← {
  x ← ⍵[1]
  xs ← ⍵[2]
  sh ← ⍴ xs
  i ← sh[1]
  j ← sh[2]
  k ← sh[3]
  (i+1) j k ⍴ (,x),(,xs)
}

matvecmul ← {
  M ← ⍺
  V ← ⍵
  sh ← ⍴M
  r ← sh[1]
  c ← sh[2]
  M2 ← r c ⍴ V
  M ← r c ⍴ M
  +/ (M × M2)
}

⍝ Backpropagation
⍝ :  network3 -> (input, output) -> ((biases2,weights2),(biases3,weights3))
backprop3 ← {
  network ← ⍺
  x ← ⍵[1]
  y ← ⍵[2]
  ⍝ Feedforward
  layer2 ← network[1]
  b2 ← layer2[1]
  w2 ← layer2[2]
  activation1 ← x
  z2 ← b2 + w2 matvecmul activation1
  activation2 ← sigmoid z2
  layer3 ← network[2]
  b3 ← layer3[1]
  w3 ← layer3[2]
  z3 ← b3 + w3 matvecmul activation2
  activation3 ← sigmoid z3
  ⍝ Backward pass
  delta3 ← (cost_derivative activation3 y) × sigmoid_prime z3
  nabla_b3 ← delta3
  nabla_w3 ← delta3 ∘.× activation2
  sp ← sigmoid_prime z2
  delta2 ← sp × (⍉w3) matvecmul delta3
  nabla_b2 ← delta2
  nabla_w2 ← delta2 ∘.× activation1
  ⍝ Return change
  nabla2 ← nabla_b2 nabla_w2
  nabla3 ← nabla_b3 nabla_w3
  nabla2 nabla3
}

⍝ Vectorized backpropagation
⍝ :  network3 -> (inputs, outputs) -> ((biases2Vec,weights2Vec),(biases3Vec,weights3Vec))
backprop3Vec ← {
  network ← ⍺            ⍝ array of biases and weights
  layer2 ← network[1]
  layer3 ← network[2]

  b2sh ← ⍴ layer2[1]
  bs2emp ← 0 (b2sh[1]) ⍴ 0.0
  w2sh ← ⍴ layer2[2]
  ws2emp ← 0 (w2sh[1]) (w2sh[2]) ⍴ 0.0
  b3sh ← ⍴ layer3[1]
  bs3emp ← 0 (b3sh[1]) ⍴ 0.0
  w3sh ← ⍴ layer3[2]
  ws3emp ← 0 (w3sh[1]) (w3sh[2]) ⍴ 0.0

  xs ← ⍵[1]
  ys ← ⍵[2]
  n ← ⊃ ⍴ xs
  ⍝ Perform backpropagation for each (x,y) pair in the mini batch
  loop ← {
    i ← ⍵[1]
    acc ← ⍵[2]
    ls2 ← acc[1]
    bs2 ← ls2[1]
    ws2 ← ls2[2]
    ls3 ← acc[2]
    bs3 ← ls3[1]
    ws3 ← ls3[2]
    nabla ← network backprop3 xs[i] ys[i]
    nabla_l2 ← nabla[1]
    nabla_b2 ← nabla_l2[1]
    nabla_w2 ← nabla_l2[2]
    nabla_l3 ← nabla[2]
    nabla_b3 ← nabla_l3[1]
    nabla_w3 ← nabla_l3[2]
    bs2 ← (vec2 nabla_b2)⍪bs2
    ws2 ← cons3 nabla_w2 ws2
    bs3 ← (vec2 nabla_b3)⍪bs3
    ws3 ← cons3 nabla_w3 ws3
    (i+1) ((bs2 ws2) (bs3 ws3))
  }
  res ← (loop ⍣ n) 1 ((bs2emp ws2emp)(bs3emp ws3emp))
  res[2]
}

update_mini_batch ← {
  ⍝ ⎕ ← 'Running minibatch'
  eta ← ⍵[1]             ⍝ learning rate
  mini_batch_xs ← ⍵[2]   ⍝ array of xs
  mini_batch_ys ← ⍵[3]   ⍝ array of ys
  n ← ⊃ ⍴ mini_batch_xs

  delta_nablas ← ⍺ backprop3Vec mini_batch_xs mini_batch_ys
  nabla ← network_sum delta_nablas
  ⍝ ⎕ ← '' ⍕ (⍴ nabla[1][2])[1]
  a ← network_sub (eta÷n) ⍺ nabla
  ⍝ ⎕ ← 'Shape of nabla2 weight.x:', ⍕ (⍴ (a[1])[2])[2]
  ⍝ ⎕ ← 'Shape of nabla2 weight.y:', ⍕ (⍴ (a[1])[2])[1]
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

multiIdx ← {
  xs ← ⍺
  n ← ≢ xs
  data ← ⍵
  sh ← 1 ↓ ⍴ data
  loop ← {
    i ← ⍵[1]
    acc ← ⍵[2]
    acc ← acc,data[i]
    (i+1) acc
  }
  r ← (loop⍣n) 1 (sh ⍴ ⍬)
  r[2]
}

SGD ← {
  training_input ← ⍵[1]
  training_output ← ⍵[2]
  network ← ⍺[1]
  epochs ← ⍺[2]
  mini_batch_size ← ⍺[3]
  eta ← ⍺[4]
  n ← ≢ training_input
  imgSz ← 28 × 28
  outSz ← 10
  batches ← ⌊ n ÷ mini_batch_size
  n ← batches × mini_batch_size          ⍝ Adjust training set
  training_input ← n ↑ training_input
  training_output ← n ↑ training_output
  loop ← {
    epoch_no ← ⍵[1]
    ⍝ ⎕ ← 'Starting Epoch'
    perm ← random_perm epoch_no n
    training_input ← perm multiIdx training_input
    training_input ← batches mini_batch_size imgSz ⍴ training_input
    training_output ← perm multiIdx training_output
    training_output ← batches mini_batch_size outSz ⍴ training_output
    f ← {
      i ← ⍵[1]
      network ← ⍵[2]
      network2 ← network update_mini_batch eta training_input[i] training_output[i]
      (i+1) network2
    }
    network ← ⍵[2]
    res ← (f⍣batches) 1 network
    (epoch_no+1) res[2]
  }
  res ← (loop⍣epochs) 1 network
  res[2]
}

test1 ← {
  x ← ⍵
  ⎕ ← sigmoid 5
  ⎕ ← sigmoid 0
  ⎕ ← sigmoid_prime 5
  ⎕ ← sigmoid_prime 0

  n ← network3_new 4 200 10
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
  training_input ← 100 28 28 ⍴ ⍳(28 × 28)
  training_output ← 100 10 ⍴ ¯1 + ⍳10
  training_data ← training_input training_output
  n ← network3_new (28 × 28) 30 10
  n ← (n epochs mini_batch_size eta) SGD training_data
  +/ (n[1])[1]
}

test2 10

⍝ test1 0
