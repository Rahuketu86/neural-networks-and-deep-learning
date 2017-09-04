---------------------------------------
-- Stochastic gradient descent learning
---------------------------------------

import "/futlib/linalg"
import "/futlib/math"

module random = import "/futlib/random"
module array  = import "/futlib/array"

module Linalg = linalg(f64)

module rng_engine = random.minstd_rand0
module ndist = random.normal_distribution f64 rng_engine

import "/futlib/radix_sort"

module pair_radix_sort = mk_radix_sort {
  type t = (i32,i32)
  let num_bits = 32
  let get_bit (bit: i32) (x:i32,_:i32) = i32((x >> bit) & 1)
}

let stddist : ndist.distribution =
  {mean=0.0f64,stddev=1.0f64}

-- The sigmoid function
let sigmoid (x:f64) =
  1.0f64 / (1.0f64 + f64.exp(-x))

-- Derivative of sigmoid
let sigmoid_prime (x:f64) =
  let s = sigmoid x
  in s * (1.0f64 - s)

-- Random numbers and random permutations
type rng = rng_engine.rng

let rand (rng:rng) (n:i32) : (rng,[n]i32) =
  let rngs = rng_engine.split_rng n rng
  let pairs = map (\rng -> rng_engine.rand rng) rngs
  let (rngs',a) = unzip pairs
  let a = map i32 a
  in (rng_engine.join_rng rngs', a)

-- [rnd_perm n] returns an array of size n containing a random permutation of iota n.
let rnd_perm (rng:rng) (n:i32) : (rng,[n]i32) =
  let (rng,a) = rand rng n
  let b = map (\x i -> (x,i)) a (iota n)
  let c = pair_radix_sort.radix_sort b
  let is = map (\(x,i) -> i) c
  in (rng,is)

let rnd_permute 't [n] (rng:rng) (a:[n]t) : (rng,[n]t) =
  let (rng,is) = rnd_perm rng n
  in unsafe(rng,map (\i -> a[i]) is)

let randn (rng:rng) (n:i32) : (rng,*[n]f64) =
  let rngs = rng_engine.split_rng n rng
  let pairs = map (\rng -> ndist.rand stddist rng) rngs
  let (rngs',a) = unzip pairs
  in (rng_engine.join_rng rngs', a)

-- Network layers

type layer [i] [j] = ([j]f64, [j][i]f64)
type layeru [i] [j] = (*[j]f64, *[j][i]f64)

let network_layer (rng:rng) (prev_sz:i32) (sz:i32) : (rng,(*[sz]f64, *[sz][prev_sz]f64)) =
  let (rng,biases) = randn rng sz --replicate sz 0.0
  let (rng,weights_flat) = randn rng (sz*prev_sz) --replicate (sz*prev_sz) 0.0
  let weights = reshape (sz,prev_sz) weights_flat
  in (rng,(biases,weights))

-- Initialise a network given a configuration (a vector of neuron
-- numbers, one number for each layer).

type network3 [i] [j] [k] = (layer [i] [j], layer [j] [k])
type network3u [i] [j] [k] = (layeru [i] [j], layeru [j] [k])

let network3 (rng:rng) (sz1:i32) (sz2:i32) (sz3:i32) : (rng,network3u [sz1] [sz2] [sz3]) =
  let (rng,layer2) = network_layer rng sz1 sz2
  let (rng,layer3) = network_layer rng sz2 sz3
  in (rng,(layer2,layer3))

let feedforward_layer [i] [j] (b:[j]f64, w:[j][i]f64) (arg:[i]f64) : [j]f64 =
  let t = Linalg.matvecmul w arg
  in map (\b t -> sigmoid (t + b)) b t

-- [(B W) feedforward3 a] returns the output of the network (B,W) given
-- the input a.
let feedforward3 [i] [j] [k] (layer2:layer[i][j],layer3:layer[j][k]) (a:[i]f64) : [k]f64 =
  let a = feedforward_layer layer2 a
  let a = feedforward_layer layer3 a
  in a

let cost_derivative [n] (output_activations:[n]f64) (y:[n]f64) : [n]f64 =
  map (-) output_activations y

let random_shuffle [n] [i] [k] (rng:rng) (training_data: [n]([i]f64,[k]f64)) : (rng,[n]([i]f64,[k]f64)) =
  rnd_permute rng training_data

let sub_network [i][j][k] (factor: f64) (network:network3[i][j][k]) (nabla:network3[i][j][k]) =
  let (l2,l3) = network
  let (b2,w2) = l2
  let (b3,w3) = l3
  let (l2n,l3n) = nabla
  let (b2n,w2n) = l2n
  let (b3n,w3n) = l3n
  let sub (b:f64) (n:f64) = b - factor*n
  let b2' = map sub b2 b2n
  let w2' = map (\x y -> map sub x y) w2 w2n
  let b3' = map sub b3 b3n
  let w3' = map (\x y -> map sub x y) w3 w3n
  in ((b2',w2'),(b3',w3'))

let outer_prod [m][n] (a:[m]f64) (b:[n]f64) : *[m][n]f64 =
  map (\x -> map (\y -> x * y) b) a

let backprop [i] [j] [k] (network:network3[i][j][k])
                         (x:[i]f64,y:[k]f64) : network3u[i][j][k] =
  -- Return a nabla (a tuple ``(nabla_b, nabla_w)``) for each (non-input)
  -- layer, which, together, represent the gradient for the cost function C_x.
  -- Feedforward
  let (layer2,layer3) = network
  let (b2,w2) = layer2
  let activation1 = x
  let z2 = map (+) (Linalg.matvecmul w2 activation1) b2
  let activation2 = map sigmoid z2
  let (b3,w3) = layer3
  let z3 = map (+) (Linalg.matvecmul w3 activation2) b3
  let activation3 = map sigmoid z3
  -- Backward pass
  let delta3 = map (*) (cost_derivative activation3 y)
                       (map sigmoid_prime z3)
  let nabla_b3 = delta3
  let nabla_w3 = outer_prod delta3 activation2
  let sp = map sigmoid_prime z2
  let delta2 = map (*) (Linalg.matvecmul (array.transpose w3) delta3) sp
  let nabla_b2 = delta2
  let nabla_w2 = outer_prod delta2 activation1
  let nabla2 = (nabla_b2,nabla_w2)
  let nabla3 = (nabla_b3,nabla_w3)
  in (nabla2,nabla3)

let layer_sum [n] [i] [j] (a: [n](layer[i][j])) : layer[i][j] =
  let (bs,ws) = unzip a
  let b = map (\xs -> reduce (+) 0f64 xs) (array.transpose bs)
  let w = map (\rs -> map (\xs -> reduce (+) 0f64 xs) rs) (rearrange (2,1,0) ws)
  in (b,array.transpose w)

let network3_sum [n] [i] [j] [k] (a: [n](network3[i][j][k])) : network3[i][j][k] =
  let (ls2,ls3) = unzip a
  in (layer_sum ls2, layer_sum ls3)

let update_mini_batch [n] [i] [j] [k] (eta:f64)
                                      (network:network3[i][j][k])
                                      (mini_batch:[n]([i]f64,[k]f64)) : network3[i][j][k] =
  -- Update the network's weights and biases by applying
  -- gradient descent using backpropagation to a single mini batch.
  -- The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
  -- is the learning rate.
  let delta_nabla = map (\d -> backprop network d) mini_batch
  let nabla = network3_sum delta_nabla
  let etadivn = eta / f64(n)
  in sub_network etadivn network nabla

let sgd [i] [j] [k] [n] (rng: rng,
                         network: network3[i][j][k],
                         training_data: [n]([i]f64,[k]f64),
                         epochs:i32,
                         mini_batch_size:i32,
                         eta:f64) : network3[i][j][k] =
  -- Train the neural network using mini-batch stochastic
  -- gradient descent.  The ``training_data`` is a list of tuples
  -- ``(x, y)`` representing the training inputs and the desired
  -- outputs.  The other non-optional parameters are
  -- self-explanatory.
  let batches = n / mini_batch_size
  let n = batches * mini_batch_size
  let training_data = training_data[0:n]
  let (_,_,network) =
    loop (rng,training_data,network) for j < epochs do
       let (a,b) = unzip training_data
       let a = reshape (batches,mini_batch_size,i) a
       let b = reshape (batches,mini_batch_size,k) b
       let network =
         loop network for x < batches do
           update_mini_batch eta network (zip a[x] b[x])
       let (rng,training_data) = random_shuffle rng training_data
       in (rng,training_data,network)
  in network

let main2() : []f64 =
  --sigmoid 5f64
  --sigmoid 0f64
  --sigmoid_prime 5f64
  --sigmoid_prime 0f64
  let rng = rng_engine.rng_from_seed [0]
  let (_, n) = network3 rng 4 200 10
  let a : [4]f64 = [3f64, 2f64, 3f64, 1f64]
  let a = feedforward3 n a
  in a

let main3 () : []f64 =
  let rng = rng_engine.rng_from_seed [0]
  let epochs = 12
  let mini_batch_size = 10
  let eta = 3.0
  let training_input = map f64 (iota 784)
  let training_output = map f64 (iota 10)
  let training_data = (training_input,training_output)
  let data = replicate 10000 training_data
  -- maybe split rng
  let (rng,n0) = network3 rng 784 30 10
  let n = sgd (rng, n0, data, epochs, mini_batch_size, eta)
  let a = feedforward3 n training_input
  in a

let convert_digit (d:i32) : [10]f64 =
  let a = replicate 10 0.0
  in unsafe(a with [d] <- 1.0)

let predict (a:[10]f64) : i32 =
  let (m,i) = reduce (\(a,i) (b,j) -> if a > b then (a,i) else (b,j))
                     (a[9],9)
                     (zip (a[:8]) (iota 9))
  in i

let main4 [m] [n] (training_imgs:[m]f64, training_results:[n]i32) : []f64 =
  let rng = rng_engine.rng_from_seed [0]
  let epochs = 1
  let mini_batch_size = 10
  let eta = 3.0
  let imgs = reshape (n, 28*28) training_imgs
  let data = map (\img d -> (img,convert_digit d)) imgs training_results
  -- split rng
  let (rng,n0) = network3 rng 784 30 10
  let n = sgd (rng, n0, data, epochs, mini_batch_size, eta)
  let training_input = imgs[232]  -- digit: 0
  let a = feedforward3 n training_input
  in a

let main [m] [n] [m2] [n2] (training_imgs:[m]f64,
                            training_results:[n]i32,
                            test_imgs:[m2]f64,
                            test_results:[n2]i32) : f64 =
  let rng = rng_engine.rng_from_seed [0]
  let epochs = 10
  let mini_batch_size = 10
  let eta = 3.0
  let imgs = reshape (n, 28*28) training_imgs
  let data = map (\img d -> (img,convert_digit d)) imgs training_results
  -- split rng
  let (rng,n0) = network3 rng 784 30 10
  let n = sgd (rng, n0, data, epochs, mini_batch_size, eta)
  let t_imgs = reshape (n2, 28*28) test_imgs
  let predictions = map (\img -> predict(feedforward3 n img)) t_imgs
  let cmps = map (\p r -> i32(p==r)) predictions test_results
  in 100.0 * f64(reduce (+) 0 cmps) / f64(n2)
