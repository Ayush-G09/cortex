import { Rng } from './rng'

export type Shape = [rows: number, cols: number]

function size(shape: Shape): number {
  return shape[0] * shape[1]
}

/** A 2D matrix (rows x cols — for this engine, always "batch x features" or a weight/bias matrix) with reverse-mode
 * automatic differentiation: every op that produces a new `Tensor` from existing ones also records how to push a
 * gradient back through itself (`_backward`) and which tensors it was built from (`_prev`). `backward()` then does
 * nothing more than a topological sort followed by calling every node's own `_backward` in reverse order — the
 * entire mechanism by which "the network learned this weight should go up" falls out of chaining a handful of
 * small, individually-obvious local derivatives (matmul's, ReLU's, softmax-cross-entropy's) rather than being
 * hand-derived layer by layer. This is the ENTIRE autodiff engine: no other file computes a gradient. */
export class Tensor {
  readonly shape: Shape
  readonly data: Float64Array
  grad: Float64Array | null = null
  readonly requiresGrad: boolean
  private _prev: Tensor[] = []
  private _backward: () => void = () => {}

  constructor(shape: Shape, data: Float64Array, requiresGrad = false) {
    if (data.length !== size(shape)) throw new Error(`data length ${data.length} does not match shape ${shape}`)
    this.shape = shape
    this.data = data
    this.requiresGrad = requiresGrad
  }

  static zeros(shape: Shape, requiresGrad = false): Tensor {
    return new Tensor(shape, new Float64Array(size(shape)), requiresGrad)
  }

  static fromArray(rows: number[][], requiresGrad = false): Tensor {
    const r = rows.length
    const c = rows[0]?.length ?? 0
    const data = new Float64Array(r * c)
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) data[i * c + j] = rows[i][j]
    return new Tensor([r, c], data, requiresGrad)
  }

  /** weights initialized N(0, scale^2) — the only place randomness enters a network's life, exactly like every
   * other project in this portfolio (world/weight creation is seeded; everything downstream is deterministic). */
  static randn(shape: Shape, rng: Rng, scale = 1, requiresGrad = true): Tensor {
    const data = new Float64Array(size(shape))
    for (let i = 0; i < data.length; i++) data[i] = rng.gaussian() * scale
    return new Tensor(shape, data, requiresGrad)
  }

  toArray(): number[][] {
    const [r, c] = this.shape
    const out: number[][] = []
    for (let i = 0; i < r; i++) out.push(Array.from(this.data.slice(i * c, i * c + c)))
    return out
  }

  private zeroGrad() {
    this.grad = new Float64Array(size(this.shape))
  }

  /** builds `out`, wiring its `_prev`/`_backward` so `backward()` can later reach every ancestor that requires a
   * gradient — a node that doesn't (and has no ancestors that do) is left alone, the same "don't compute gradients
   * nothing needs" pruning every real autodiff engine does. */
  private static makeResult(shape: Shape, data: Float64Array, parents: Tensor[], backward: () => void): Tensor {
    const needsGrad = parents.some((p) => p.requiresGrad)
    const out = new Tensor(shape, data, needsGrad)
    if (needsGrad) {
      out._prev = parents
      out._backward = backward
    }
    return out
  }

  add(other: Tensor): Tensor {
    const [r, c] = this.shape
    // supports exact-shape addition and bias broadcasting: other is [1, c] added to every row of [r, c]
    const broadcastBias = other.shape[0] === 1 && other.shape[1] === c && r !== 1
    if (!broadcastBias && (other.shape[0] !== r || other.shape[1] !== c)) {
      throw new Error(`shape mismatch in add: ${this.shape} vs ${other.shape}`)
    }
    const data = new Float64Array(r * c)
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        data[i * c + j] = this.data[i * c + j] + (broadcastBias ? other.data[j] : other.data[i * c + j])
      }
    }
    const out = Tensor.makeResult([r, c], data, [this, other], () => {
      if (this.requiresGrad) {
        if (!this.grad) this.zeroGrad()
        for (let k = 0; k < data.length; k++) this.grad![k] += out.grad![k]
      }
      if (other.requiresGrad) {
        if (!other.grad) other.zeroGrad()
        if (broadcastBias) {
          for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) other.grad![j] += out.grad![i * c + j]
        } else {
          for (let k = 0; k < data.length; k++) other.grad![k] += out.grad![k]
        }
      }
    })
    return out
  }

  matmul(other: Tensor): Tensor {
    const [m, k] = this.shape
    const [k2, n] = other.shape
    if (k !== k2) throw new Error(`shape mismatch in matmul: ${this.shape} vs ${other.shape}`)
    const data = new Float64Array(m * n)
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const a = this.data[i * k + p]
        if (a === 0) continue
        for (let j = 0; j < n; j++) data[i * n + j] += a * other.data[p * n + j]
      }
    }
    const out = Tensor.makeResult([m, n], data, [this, other], () => {
      // dL/dA = dL/dOut @ B^T, dL/dB = A^T @ dL/dOut — the two standard matmul-gradient identities
      if (this.requiresGrad) {
        if (!this.grad) this.zeroGrad()
        for (let i = 0; i < m; i++) {
          for (let p = 0; p < k; p++) {
            let s = 0
            for (let j = 0; j < n; j++) s += out.grad![i * n + j] * other.data[p * n + j]
            this.grad![i * k + p] += s
          }
        }
      }
      if (other.requiresGrad) {
        if (!other.grad) other.zeroGrad()
        for (let p = 0; p < k; p++) {
          for (let j = 0; j < n; j++) {
            let s = 0
            for (let i = 0; i < m; i++) s += this.data[i * k + p] * out.grad![i * n + j]
            other.grad![p * n + j] += s
          }
        }
      }
    })
    return out
  }

  relu(): Tensor {
    const data = new Float64Array(this.data.length)
    for (let i = 0; i < data.length; i++) data[i] = Math.max(0, this.data[i])
    const out = Tensor.makeResult(this.shape, data, [this], () => {
      if (!this.requiresGrad) return
      if (!this.grad) this.zeroGrad()
      for (let i = 0; i < data.length; i++) this.grad![i] += (this.data[i] > 0 ? 1 : 0) * out.grad![i]
    })
    return out
  }

  /** row-wise softmax — used standalone for inference (turning logits into a probability distribution to display),
   * kept separate from the fused loss below because a fused softmax+cross-entropy is what you want for TRAINING
   * (numerically stable, simpler gradient), while plain softmax is what you want for just reading probabilities
   * off a trained network. */
  softmax(): Tensor {
    const [r, c] = this.shape
    const data = new Float64Array(r * c)
    for (let i = 0; i < r; i++) {
      let max = -Infinity
      for (let j = 0; j < c; j++) max = Math.max(max, this.data[i * c + j])
      let sum = 0
      for (let j = 0; j < c; j++) {
        const e = Math.exp(this.data[i * c + j] - max)
        data[i * c + j] = e
        sum += e
      }
      for (let j = 0; j < c; j++) data[i * c + j] /= sum
    }
    return new Tensor([r, c], data, false) // inference-only; not wired into the training graph
  }

  /** Fused softmax + cross-entropy against integer class labels (one per row): computes the mean loss over the
   * batch as a 1x1 tensor, AND gives that 1x1 tensor a `_backward` whose gradient w.r.t. the logits is the classic,
   * beautifully simple `(softmax(logits) - one_hot(labels)) / batchSize` — the reason softmax and cross-entropy are
   * almost always fused in real frameworks: computing them separately is both less numerically stable (the log of
   * a near-zero softmax output) and needlessly more complex to differentiate through. */
  softmaxCrossEntropy(labels: number[]): Tensor {
    const [r, c] = this.shape
    if (labels.length !== r) throw new Error(`expected ${r} labels, got ${labels.length}`)
    const probs = new Float64Array(r * c)
    let loss = 0
    for (let i = 0; i < r; i++) {
      let max = -Infinity
      for (let j = 0; j < c; j++) max = Math.max(max, this.data[i * c + j])
      let sum = 0
      for (let j = 0; j < c; j++) {
        const e = Math.exp(this.data[i * c + j] - max)
        probs[i * c + j] = e
        sum += e
      }
      for (let j = 0; j < c; j++) probs[i * c + j] /= sum
      const label = labels[i]
      loss += -Math.log(Math.max(probs[i * c + label], 1e-12))
    }
    loss /= r

    const out = Tensor.makeResult([1, 1], Float64Array.of(loss), [this], () => {
      if (!this.requiresGrad) return
      if (!this.grad) this.zeroGrad()
      const upstream = out.grad![0] // usually 1 (loss.backward() with no further consumer), but respected anyway
      for (let i = 0; i < r; i++) {
        for (let j = 0; j < c; j++) {
          const target = j === labels[i] ? 1 : 0
          this.grad![i * c + j] += ((probs[i * c + j] - target) / r) * upstream
        }
      }
    })
    return out
  }

  /** Reverse-mode autodiff: topologically sorts every ancestor this tensor was built from, seeds this tensor's own
   * gradient to 1 (the standard "d(loss)/d(loss) = 1" starting point — `backward()` is meant to be called on a
   * scalar, i.e. a 1x1 tensor, exactly the shape `softmaxCrossEntropy` produces), then runs each node's own
   * `_backward` in reverse topological order so every gradient is fully accumulated before it's consumed. */
  backward(): void {
    if (size(this.shape) !== 1) throw new Error('backward() can only be called on a scalar (1x1) tensor')
    const topo: Tensor[] = []
    const visited = new Set<Tensor>()
    const visit = (t: Tensor) => {
      if (visited.has(t)) return
      visited.add(t)
      for (const p of t._prev) visit(p)
      topo.push(t)
    }
    visit(this)

    this.zeroGrad()
    this.grad![0] = 1
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward()
  }
}
