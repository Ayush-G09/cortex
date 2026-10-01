import { describe, expect, it } from 'vitest'
import { Tensor } from './tensor'
import { Rng } from './rng'

describe('Tensor — basic ops produce correct values', () => {
  it('add: elementwise', () => {
    const a = Tensor.fromArray([[1, 2], [3, 4]])
    const b = Tensor.fromArray([[10, 20], [30, 40]])
    expect(a.add(b).toArray()).toEqual([[11, 22], [33, 44]])
  })

  it('add: broadcasts a [1,c] bias across every row of a [r,c] tensor', () => {
    const a = Tensor.fromArray([[1, 2], [3, 4], [5, 6]])
    const bias = Tensor.fromArray([[100, 200]])
    expect(a.add(bias).toArray()).toEqual([[101, 202], [103, 204], [105, 206]])
  })

  it('matmul: standard matrix multiplication', () => {
    const a = Tensor.fromArray([[1, 2, 3], [4, 5, 6]]) // 2x3
    const b = Tensor.fromArray([[7, 8], [9, 10], [11, 12]]) // 3x2
    // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    expect(a.matmul(b).toArray()).toEqual([[58, 64], [139, 154]])
  })

  it('relu: zeroes negatives, passes positives through unchanged', () => {
    const a = Tensor.fromArray([[-2, 0, 3], [-0.5, 5, -9]])
    expect(a.relu().toArray()).toEqual([[0, 0, 3], [0, 5, 0]])
  })

  it('softmax: each row sums to 1 and is monotonic with the logits', () => {
    const a = Tensor.fromArray([[1, 2, 3], [0, 0, 0]])
    const s = a.softmax().toArray()
    for (const row of s) expect(row.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 10)
    expect(s[0][2]).toBeGreaterThan(s[0][1])
    expect(s[0][1]).toBeGreaterThan(s[0][0])
    expect(s[1][0]).toBeCloseTo(1 / 3, 10)
  })

  it('softmaxCrossEntropy: a confident correct prediction has much lower loss than a confident wrong one', () => {
    const correct = Tensor.fromArray([[10, 0, 0]]).softmaxCrossEntropy([0]).data[0]
    const wrong = Tensor.fromArray([[10, 0, 0]]).softmaxCrossEntropy([1]).data[0]
    expect(correct).toBeLessThan(wrong)
    expect(correct).toBeLessThan(0.01)
  })
})

describe('Tensor — backward() matches values computed by hand', () => {
  it('matmul + sum-style scalar: gradient of A@B loss w.r.t. A and B', () => {
    // f = sum(A @ B), A is 2x2, B is 2x2 — d f/d A = B summed across columns broadcast into rows (ones @ B^T),
    // d f/d B = A^T @ ones. Easiest to just check against the closed form for this specific tiny case.
    const a = Tensor.fromArray([[1, 2], [3, 4]], true)
    const b = Tensor.fromArray([[5, 6], [7, 8]], true)
    const out = a.matmul(b) // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    // sum every element of `out` to a true 1x1 scalar via matmul with ones vectors on both sides
    const onesCol = Tensor.fromArray([[1], [1]])
    const rowSums = out.matmul(onesCol) // 2x1, each row's sum
    const onesRow = Tensor.fromArray([[1, 1]])
    const total = onesRow.matmul(rowSums) // 1x1 = grand total
    total.backward()

    // d(total)/d(out) is all-ones (2x2); d(out)/dA = B^T-style contraction => dA = ones(2x2) @ B^T
    // B^T = [[5,7],[6,8]]; ones(2x2) @ B^T = [[5+6,7+8],[5+6,7+8]] = [[11,15],[11,15]]
    expect(Array.from(a.grad!)).toEqual([11, 15, 11, 15])
    // dB = A^T @ ones(2x2); A^T = [[1,3],[2,4]]; A^T @ ones(2x2) = [[4,4],[6,6]]
    expect(Array.from(b.grad!)).toEqual([4, 4, 6, 6])
  })

  it('a tensor used twice in the graph accumulates gradients from both uses (diamond graph)', () => {
    const x = Tensor.fromArray([[2, 3]], true)
    const onesCol = Tensor.fromArray([[1], [1]])
    // total = sum(x) + sum(x) = 2*sum(x) -> d(total)/dx = [2, 2]
    const sumX = x.matmul(onesCol) // 1x1
    const total = sumX.add(sumX)
    total.backward()
    expect(Array.from(x.grad!)).toEqual([2, 2])
  })
})

describe('Tensor — backward() matches numerical (finite-difference) gradients', () => {
  // The decisive proof for any autodiff engine: perturb each input element by a tiny epsilon, recompute the
  // scalar output, and check the resulting slope matches what backward() computed analytically. If these two
  // completely independent ways of computing "how loss changes" agree to several decimal places across a chain of
  // matmul -> add -> relu -> matmul -> add -> softmaxCrossEntropy, the whole engine is doing real calculus, not
  // something that merely happens to train.
  function numericalGrad(build: (w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) => Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, target: Tensor): Float64Array {
    const eps = 1e-5
    const grad = new Float64Array(target.data.length)
    for (let i = 0; i < target.data.length; i++) {
      const orig = target.data[i]
      target.data[i] = orig + eps
      const lossPlus = build(w1, b1, w2, b2).data[0]
      target.data[i] = orig - eps
      const lossMinus = build(w1, b1, w2, b2).data[0]
      target.data[i] = orig
      grad[i] = (lossPlus - lossMinus) / (2 * eps)
    }
    return grad
  }

  it('gradients through a full 2-layer MLP + softmax-cross-entropy match finite differences', () => {
    const rng = new Rng(7)
    const x = Tensor.fromArray([[0.5, -1.2, 0.3], [1.1, 0.2, -0.7]]) // batch of 2, 3 features
    const labels = [1, 0]
    const w1 = Tensor.randn([3, 4], rng, 0.5, true)
    const b1 = Tensor.zeros([1, 4], true)
    const w2 = Tensor.randn([4, 2], rng, 0.5, true)
    const b2 = Tensor.zeros([1, 2], true)

    function forward(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor): Tensor {
      const h = x.matmul(w1).add(b1).relu()
      const logits = h.matmul(w2).add(b2)
      return logits.softmaxCrossEntropy(labels)
    }

    const loss = forward(w1, b1, w2, b2)
    loss.backward()

    for (const [name, t] of [['w1', w1], ['b1', b1], ['w2', w2], ['b2', b2]] as const) {
      const numeric = numericalGrad(forward, w1, b1, w2, b2, t)
      for (let i = 0; i < t.data.length; i++) {
        expect(t.grad![i], `${name}[${i}]`).toBeCloseTo(numeric[i], 4)
      }
    }
  })
})
