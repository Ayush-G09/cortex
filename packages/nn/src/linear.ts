import { Rng, Tensor } from '@cortex/tensor'

/** One fully-connected layer: `y = x @ W + b`. Weights are He-initialized (`N(0, 2/fanIn)`), the standard choice
 * for layers followed by ReLU — it keeps activation variance roughly stable layer to layer instead of vanishing or
 * exploding as depth increases, purely a matter of the initial random draw (training still has to do the actual
 * learning; a good init just keeps it from starting somewhere pathological). */
export class Linear {
  readonly weight: Tensor
  readonly bias: Tensor

  constructor(inFeatures: number, outFeatures: number, rng: Rng) {
    const scale = Math.sqrt(2 / inFeatures)
    this.weight = Tensor.randn([inFeatures, outFeatures], rng, scale, true)
    this.bias = Tensor.zeros([1, outFeatures], true)
  }

  forward(x: Tensor): Tensor {
    return x.matmul(this.weight).add(this.bias)
  }

  params(): Tensor[] {
    return [this.weight, this.bias]
  }
}
