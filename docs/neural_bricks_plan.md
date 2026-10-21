# Neural bricks for online and classic learning: the plan and its design choices

Dense-Armor streams one sample at a time. The next set of modules adds neural bricks, and every brick has two modes that share the same code: a batch or mini-batch (classic) and one sample with `learn_one` (online). This page lists the seven bricks and the design choices written down before the first line of code, each with the paper page it rests on.

## The seven bricks

| # | Brick | Content |
|---|---|---|
| N01 | Foundations | parameters that hold their gradient, He and Xavier initialisations, activations and losses with `forward` and `backward`, a gradient check against finite differences |
| N02 | Layers | dense, convolution, pooling, recurrent (RNN, LSTM, GRU), embedding, batch and layer normalisation, dropout, residual connections |
| N03 | Chains and architectures | `Sequential` and layer graphs; MLP, CNN, ResNet, recurrent network and transformer block built from the same pieces |
| N04 | Standard training | epochs, shuffling, early stopping, learning-rate schedules, regularisation, evaluation; the optimizers SGD, AdaGrad, Adam, FTRL and Online Newton Step already in the library |
| N05 | Continuous training | the same model trained one sample at a time without blocking the main program, with a p99 latency budget, checkpoints and protection |
| N06 | Encoders and decoders | modules that compress a stream (for example video thinned by incremental PCA) or expand it into a prediction |
| N07 | Composable attention | single-head, multi-head, self and cross attention, and a gate that reads the video only when the audio crosses a threshold |

## Design choices of N01

Each choice is a decision of this code where the papers give no rule. The papers are in the project index with page numbers.

| Choice | Value | Rests on |
|---|---|---|
| He initialisation | zero-mean Gaussian, standard deviation `sqrt(2 / ((1 + a^2) * fan_in))` | He et al. 2015, Eqn.(10) page 4 and Eqn.(15) page 5 |
| Xavier initialisation | the case `a = 1` of the same equation, standard deviation `sqrt(1 / fan_in)` | He et al. 2015, page 5: with `a = 1` the equation is the linear case of Glorot and Bengio |
| Parametric rectifier | `f(y) = y` for `y > 0`, `a*y` otherwise; `a = 0` is ReLU | He et al. 2015, Eqn.(1) page 2 |
| Derivative of ReLU at 0 | 0 | choice of this code |
| Gradient check | central differences in 64-bit floats inside a local `jax.enable_x64(True)` block, `eps = 1e-6`; the precision outside the block stays 32 bit | choice of this code |
| Cross-entropy and binary cross-entropy | log-sum-exp form and `max(z, 0) - z*t + log(1 + exp(-|z|))`, finite at `z = 1000` | choice of this code |
| Arrays | `jax.numpy`, pure `forward` and `backward` functions that work under `jax.jit` | the library runs on JAX |
| Batch and one sample | a batch of one sample equals the matching row of a larger batch | the online and classic modes share the code |

## Details

### References
- He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv:1502.01852. Page 2 (Eqn.(1)), page 4 (Eqn.(10)), page 5 (Eqn.(15)).

### Development diary
The plan is written; the code of N01 is in progress. The Huber loss and the uniform range of the Glorot initialisation wait for their source papers.
