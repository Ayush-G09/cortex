import { Rng, Tensor } from '@cortex/tensor'
import { Linear } from './linear'

/** A plain feedforward classifier: alternating `Linear` layers and ReLU, EXCEPT the final layer, which is left as
 * raw logits — softmax belongs fused into the loss during training (`Tensor.softmaxCrossEntropy`) and applied
 * standalone only at inference time (`Tensor.softmax`), never baked into the network itself. */
export class MLP {
  readonly layers: Linear[]

  constructor(sizes: number[], rng: Rng) {
    if (sizes.length < 2) throw new Error('MLP needs at least an input and an output size')
    this.layers = []
    for (let i = 0; i < sizes.length - 1; i++) this.layers.push(new Linear(sizes[i], sizes[i + 1], rng))
  }

  forward(x: Tensor): Tensor {
    let h = x
    for (let i = 0; i < this.layers.length; i++) {
      h = this.layers[i].forward(h)
      if (i < this.layers.length - 1) h = h.relu()
    }
    return h
  }

  params(): Tensor[] {
    return this.layers.flatMap((l) => l.params())
  }

  /** flat weights, for saving a trained network out to JSON (a demo needs the numbers, not the JS objects) */
  toWeights(): number[][] {
    return this.params().map((p) => Array.from(p.data))
  }

  loadWeights(weights: number[][]): void {
    const params = this.params()
    if (weights.length !== params.length) throw new Error(`expected ${params.length} parameter tensors, got ${weights.length}`)
    params.forEach((p, i) => {
      if (weights[i].length !== p.data.length) throw new Error(`parameter ${i} length mismatch`)
      p.data.set(weights[i])
    })
  }
}
