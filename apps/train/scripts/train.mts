/**
 * Trains the from-scratch MLP (784 -> 64 -> 10, ReLU hidden, softmax-cross-entropy loss) on the real MNIST subset
 * saved by prepare-data.mts, via real mini-batch SGD through the hand-written autodiff engine in @cortex/tensor —
 * no PyTorch, no TensorFlow, nothing that computes a gradient except this repo's own code.
 *
 * Proof, same standard as every other project in this portfolio: the trained network's test accuracy is compared
 * against a majority-class baseline (predict "0" for every image — right exactly 1/10 of the time on this balanced
 * subset) AND against the SAME untrained architecture (random weights, zero training) — so "the network learned
 * digit recognition" is a comparison against two different kinds of "didn't actually learn," not a bare number.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { Rng, Tensor } from '@cortex/tensor'
import { MLP, SGD } from '@cortex/nn'

interface RawData {
  trainCount: number
  testCount: number
  trainImages: string
  trainLabels: number[]
  testImages: string
  testLabels: number[]
}

const HIDDEN_SIZE = 64
const BATCH_SIZE = 32
const EPOCHS = 18
const LR = 0.5
const SEED = 20260101

function decodeImages(base64: string, count: number): Float64Array {
  const bytes = Buffer.from(base64, 'base64')
  const out = new Float64Array(count * 784)
  for (let i = 0; i < out.length; i++) out[i] = bytes[i] / 255
  return out
}

function batchTensor(images: Float64Array, indices: number[]): Tensor {
  const data = new Float64Array(indices.length * 784)
  indices.forEach((idx, row) => data.set(images.subarray(idx * 784, idx * 784 + 784), row * 784))
  return new Tensor([indices.length, 784], data)
}

function accuracy(mlp: MLP, images: Float64Array, labels: number[], count: number): number {
  const x = batchTensor(images, Array.from({ length: count }, (_, i) => i))
  const probs = mlp.forward(x).softmax().toArray()
  let correct = 0
  for (let i = 0; i < count; i++) {
    const pred = probs[i].indexOf(Math.max(...probs[i]))
    if (pred === labels[i]) correct++
  }
  return correct / count
}

function majorityBaselineAccuracy(trainLabels: number[], testLabels: number[]): number {
  const counts = new Array(10).fill(0)
  for (const l of trainLabels) counts[l]++
  const majorityClass = counts.indexOf(Math.max(...counts))
  return testLabels.filter((l) => l === majorityClass).length / testLabels.length
}

function main() {
  console.log('Loading MNIST subset...')
  const raw: RawData = JSON.parse(readFileSync(new URL('../data/mnist.json', import.meta.url), 'utf-8'))
  const trainImages = decodeImages(raw.trainImages, raw.trainCount)
  const testImages = decodeImages(raw.testImages, raw.testCount)

  const rng = new Rng(SEED)
  const mlp = new MLP([784, HIDDEN_SIZE, 10], rng)
  const opt = new SGD(mlp.params(), LR)

  const untrainedAccuracy = accuracy(mlp, testImages, raw.testLabels, raw.testCount)
  const baselineAccuracy = majorityBaselineAccuracy(raw.trainLabels, raw.testLabels)
  console.log(`Untrained network test accuracy: ${(untrainedAccuracy * 100).toFixed(1)}%`)
  console.log(`Majority-class baseline accuracy: ${(baselineAccuracy * 100).toFixed(1)}%`)

  const history: { epoch: number; loss: number; testAccuracy: number }[] = []
  const t0 = Date.now()

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Fisher-Yates shuffle of training indices, seeded — a fresh, deterministic batch order every epoch
    const order = Array.from({ length: raw.trainCount }, (_, i) => i)
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }

    let epochLoss = 0
    let batchCount = 0
    for (let start = 0; start < order.length; start += BATCH_SIZE) {
      const batchIdx = order.slice(start, start + BATCH_SIZE)
      const x = batchTensor(trainImages, batchIdx)
      const labels = batchIdx.map((i) => raw.trainLabels[i])

      opt.zeroGrad()
      const logits = mlp.forward(x)
      const loss = logits.softmaxCrossEntropy(labels)
      loss.backward()
      opt.step()

      epochLoss += loss.data[0]
      batchCount++
    }

    const testAcc = accuracy(mlp, testImages, raw.testLabels, raw.testCount)
    history.push({ epoch, loss: epochLoss / batchCount, testAccuracy: testAcc })
    console.log(`epoch ${epoch + 1}/${EPOCHS}  loss ${(epochLoss / batchCount).toFixed(4)}  test acc ${(testAcc * 100).toFixed(1)}%`)
  }

  const elapsed = (Date.now() - t0) / 1000
  const finalAccuracy = history[history.length - 1].testAccuracy
  console.log(`\nDone in ${elapsed.toFixed(1)}s.`)
  console.log(`Final test accuracy: ${(finalAccuracy * 100).toFixed(1)}%`)
  console.log(`  vs untrained network: ${(untrainedAccuracy * 100).toFixed(1)}%`)
  console.log(`  vs majority-class baseline: ${(baselineAccuracy * 100).toFixed(1)}%`)

  const output = {
    config: { hiddenSize: HIDDEN_SIZE, batchSize: BATCH_SIZE, epochs: EPOCHS, lr: LR, seed: SEED },
    untrainedAccuracy,
    baselineAccuracy,
    history,
    shape: [784, HIDDEN_SIZE, 10],
    weights: mlp.toWeights(),
  }

  mkdirSync(new URL('../../web/public', import.meta.url), { recursive: true })
  const outPath = new URL('../../web/public/model.json', import.meta.url)
  writeFileSync(outPath, JSON.stringify(output))
  console.log(`Wrote ${outPath.pathname.replace(/^\//, '')}`)
}

main()
