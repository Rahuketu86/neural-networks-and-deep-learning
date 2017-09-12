println ← {
  ⍵ ⍝ ⎕ ← ⍵
}

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
  j ← i×3
  k ← i×11
  n ← ⍵[2]
  xs ← ⍳n
  xs ← (i ⌽ j ↓ xs), j ↑ xs
  (j ⌽ k ↓ xs), k ↑ xs
}

hash ← {
  v ← ⍵
  v ← 73244475 × v xor v srl 16
  v ← 73244475 × v xor v srl 16
  v ← v xor v srl 16
  v
}

rand_a ← 48271
rand_c ← 0
rand_m ← 2147483647

randi ← {
  x ← ⌊ ⍵
  y ← rand_m | ((rand_a × x) + rand_c)
  y
}

split_rng ← {
  x ← ⌊⍺
  n ← ⍵
  { rand_m | x xor hash ⍵ }¨⍳n
}

⍝ randr ← {
⍝   x ← randi ⍵
⍝   y ← x ÷ rand_m
⍝   x y
⍝ }

⍝ randr_vec ← {
⍝   rngs ← ⍺ split_rng ⍵
⍝   rng ← join_rng rngs
⍝   vs ← { ⍵ ÷ rand_m}¨rngs
⍝   rng vs
⍝ }

bigint ← 1000000
rand ← {
  ⍵
  (? bigint) ÷ bigint
}

⍝ Random normal distribution
randn_vec ← {
  join_rng ← {
    rand_m | xor/⍵
  }
  rngs ← ⍺ split_rng 2×⍵
  rng ← join_rng rngs
  rnds ← rngs ÷ rand_m
  rnds1 ← ⍵ ↑ rnds
  rnds2 ← ⍵ ↓ rnds
  rndn ← {
    r ← (¯2 × ⍟ ⍺ ) * 0.5
    theta ← ⍵ × ○2
    r × 2○ theta
  }
  rs ← rnds1 rndn¨ rnds2
  rng rs
}

randn_vec_interp ← {
  ⍺
  randn ← {
    ⍵
    u1 ← (?bigint) ÷ bigint
    u2 ← (?bigint) ÷ bigint
    r ← (¯2 × ⍟ u1) * 0.5
    theta ← u2 × ○2
    r × 2○ theta
  }
  ⍺ (randn ¨⍳ ⍵)

}

⍝ Initialise a network layer.
⍝ [prev_sz layer_new sz] returns an initialized network layer
⍝ (B,W) of biases and weights, where B is a vector of size sz and W is a
⍝ matrix of dimension prev_sz × sz.
layer_new ← {
  rng ← ⍺
  prev_sz ← ⍵[1]
  sz ← ⍵[2]
  rng_rs ← rng randn_vec sz + sz × prev_sz
  rng ← rng_rs[1]
  rs ← rng_rs[2]
  biases ← sz ↑ rs
  weights ← sz prev_sz ⍴ sz ↓ rs
  rng (biases weights)
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

pack_layer ← {
  b ← ⍵[1] ⋄ w ← ⍵[2]
  b,,w
}

pack ← {
  (pack_layer ⍵[1]),pack_layer ⍵[2]
}

⍝ unpack_layer ← {
⍝   b ← ⍺[2] ↑ ⍵
⍝   ⍵ ← ⍺[2] ↓ ⍵
⍝   w ← (⍺[2]) (⍺[1]) ⍴ (⍺[2]×⍺[1]) ↑ ⍵
⍝   b w
⍝ }

⍝ unpack ← {
⍝   l2 ← ⍺[1] ⍺[2] unpack_layer ⍵
⍝   ⍵ ← (⍺[2]×⍺[1]+1) ↓ ⍵
⍝   l3 ← ⍺[2] ⍺[3] unpack_layer ⍵
⍝   l2 l3
⍝ }

⍝ Initialise a network given a configuration (a vector of neuron
⍝ numbers, one number for each layer).
network3_new ← {
  sz1 ← ⍵[1] ⋄ sz2 ← ⍵[2] ⋄ sz3 ← ⍵[3]
  rng ← 1
  rng_layer2 ← rng layer_new sz1 sz2
  rng ← rng_layer2[1]
  layer2 ← rng_layer2[2]
  rng_layer3 ← rng layer_new sz2 sz3
  layer3 ← rng_layer3[2]
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
  layer3 layer_feedforward (layer2 layer_feedforward ⍵)
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

⍝ Turn a digit into a 10d unit vector
from_digit ← { ⍵=¯1+⍳10 }

⍝ Predict a digit based on the output
⍝ layer's activation vector
predict_digit ← { ¯1++/(⍳≢⍵)×⍵=⌈/⍵ }

⍝ Backpropagation
⍝ :  network3 -> (input, output) -> ((biases2,weights2),(biases3,weights3))
backprop3 ← {
  network ← ⍺
  x ← ⍵[1]
  y ← from_digit ⍵[2]
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
⍝ :  network3 -> {batch_size:int,batch_no:int,perm:[]int, inputs, outputs} -> ((biases2Vec,weights2Vec),(biases3Vec,weights3Vec))
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

  batch_size ← ⍵[1]
  batch_no ← ⍵[2]
  perm ← ⍵[3]
  training_input ← ⍵[4]
  training_outputs ← ⍵[5]
  ⍝ Perform backpropagation for each (x,y) pair in the batch
  loop ← {
    i ← ⍵[1]
    idx ← perm[i+batch_size×batch_no-1]
    acc ← ⍵[2]
    ls2 ← acc[1]
    bs2 ← ls2[1]
    ws2 ← ls2[2]
    ls3 ← acc[2]
    bs3 ← ls3[1]
    ws3 ← ls3[2]
    nabla ← network backprop3 training_input[idx] training_outputs[idx]
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
  res ← (loop ⍣ batch_size) 1 ((bs2emp ws2emp)(bs3emp ws3emp))
  res[2]
}

update_batch ← {
  ⍝ ⎕ ← 'Running batch'
  batch_size ← ⍵[1]       ⍝ size of a batch
  batch_no ← ⍵[2]         ⍝ batch number
  perm ← ⍵[3]             ⍝ data permutation
  eta ← ⍵[4]              ⍝ learning rate
  training_input ← ⍵[5]   ⍝ array of xs
  training_outputs ← ⍵[6] ⍝ array of ys

  delta_nablas ← ⍺ backprop3Vec batch_size batch_no perm training_input training_outputs
  nabla ← network_sum delta_nablas
  ⍝ ⎕ ← '' ⍕ (⍴ nabla[1][2])[1]
  a ← network_sub (eta÷batch_size) ⍺ nabla
  ⍝ ⎕ ← 'Shape of nabla2 weight.x:', ⍕ (⍴ (a[1])[2])[2]
  ⍝ ⎕ ← 'Shape of nabla2 weight.y:', ⍕ (⍴ (a[1])[2])[1]
  a
}

⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝
⍝ Stochastic Gradient Descent
⍝
⍝ [(network epochs batch_size eta) SGD training_data] returns a
⍝ new network trained with the provided training data. The input `epochs`
⍝ specifies the number of random shuffes of the training data. The input
⍝ `batch_size` specifies the training data split size in each epoch. The
⍝ input `eta` is the training rate.
⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝ ⍝

⍝ Stochastic gradient decent
SGD ← {
  training_input ← ⍵[1]
  training_output ← ⍵[2]
  network ← ⍺[1]
  epochs ← ⍺[2]
  batch_size ← ⍺[3]
  eta ← ⍺[4]
  n ← ≢ training_input
  batches ← ⌊ n ÷ batch_size
  n ← batches × batch_size                 ⍝ Adjust training set
  training_input ← n ↑ training_input
  training_output ← n ↑ training_output
  loop ← {
    epoch_no ← ⍵[1]
    println 'Starting Epoch'
    perm ← random_perm epoch_no n
    f ← {
      println 'Starting batch'
      i ← ⍵[1]
      network ← ⍵[2]
      network2 ← network update_batch (batch_size i perm eta training_input training_output)
      (i+1) network2
    }
    network ← ⍵[2]
    res ← (f⍣batches) 1 network
    (epoch_no+1) res[2]
  }
  res ← (loop⍣epochs) 1 network
  res[2]
}

run ← {
  epochs ← 10
  batch_size ← 20
  eta ← 0.5
  training_inputs ← 50000 (28×28) ⍴ ⍵[1]    ⍝ 50000 images, each being a flat float vector
  training_results ← 50000 ⍴ ⍵[2]           ⍝ 50000 integers, each denoting a digit
  n ← network3_new (28×28) 30 10
  n ← (n epochs batch_size eta) SGD (training_inputs training_results)
  test_inputs ← 10000 (28×28) ⍴ ⍵[3]
  test_results ← 10000 ⍴ ⍵[4]
  res ← {
   in ← test_inputs[⍵]
   out ← n feedforward3 in
   v ← predict_digit out
   expected ← test_results[⍵]
   v = expected
  }¨⍳≢test_inputs
  rate ← (+/res)÷≢test_inputs
  100 × rate
}


println 'reading training data'
training_imgs ← ReadCSVDouble '../futhark/mnist_training_small_input.apl.txt'
training_results ← ReadCSVInt '../futhark/mnist_training_small_results.apl.txt'

println 'reading test data'
test_imgs ← ReadCSVDouble '../futhark/mnist_test_small_input.apl.txt'
test_results ← ReadCSVInt '../futhark/mnist_test_small_results.apl.txt'
println 'done reading data'

run training_imgs training_results test_imgs test_results

⍝ test1 0
