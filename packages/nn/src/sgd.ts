import { Tensor } from '@cortex/tensor'

/** Plain mini-batch stochastic gradient descent: `param -= lr * grad`, nothing more — no momentum, no Adam. This
 * is deliberately the simplest possible optimizer, because the point of this project is proving the from-scratch
 * autodiff + training LOOP works, not chasing state-of-the-art convergence speed. */
export class SGD {
  constructor(private params: Tensor[], private lr: number) {}

  step(): void {
    for (const p of this.params) {
      if (!p.grad) continue
      for (let i = 0; i < p.data.length; i++) p.data[i] -= this.lr * p.grad[i]
    }
  }

  /** must be called before each backward() — gradients ACCUMULATE across calls (that's what makes the diamond-graph
   * case in the autodiff engine work at all), so a fresh mini-batch needs a fresh (zeroed) grad buffer first. */
  zeroGrad(): void {
    for (const p of this.params) {
      if (p.grad) p.grad.fill(0)
    }
  }
}
