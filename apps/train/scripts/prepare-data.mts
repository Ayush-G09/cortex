/**
 * Draws a real MNIST subset (via the `mnist` npm package, which bundles the actual 60k/10k handwritten-digit
 * dataset as plain JS, so no manual gzip/idx parsing is needed) and saves it to a single compact JSON file that
 * `train.mts` and the test suite both read — so "the exact data a run trained on" is a saved, inspectable artifact,
 * not re-drawn (and therefore silently different) every time.
 *
 * Pixel values are quantized from the library's [0,1] floats back to bytes (0-255) and base64-encoded, which
 * shrinks the file by roughly 8x versus a JSON array of floats — the difference between a multi-hundred-MB file
 * and one small enough to keep alongside the code.
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import mnist from 'mnist'

const TRAIN_COUNT = 6000
const TEST_COUNT = 1500

function toBytes(samples: { input: number[]; output: number[] }[]): { bytes: Uint8Array; labels: number[] } {
  const bytes = new Uint8Array(samples.length * 784)
  const labels = new Array<number>(samples.length)
  samples.forEach((s, i) => {
    for (let j = 0; j < 784; j++) bytes[i * 784 + j] = Math.round(s.input[j] * 255)
    labels[i] = s.output.indexOf(1)
  })
  return { bytes, labels }
}

function main() {
  console.log(`Drawing ${TRAIN_COUNT} training + ${TEST_COUNT} test MNIST digits...`)
  const data = mnist.set(TRAIN_COUNT, TEST_COUNT)
  const train = toBytes(data.training)
  const test = toBytes(data.test)

  const output = {
    trainCount: TRAIN_COUNT,
    testCount: TEST_COUNT,
    trainImages: Buffer.from(train.bytes).toString('base64'),
    trainLabels: train.labels,
    testImages: Buffer.from(test.bytes).toString('base64'),
    testLabels: test.labels,
  }

  mkdirSync(new URL('../data', import.meta.url), { recursive: true })
  const outPath = new URL('../data/mnist.json', import.meta.url)
  writeFileSync(outPath, JSON.stringify(output))
  console.log(`Wrote ${outPath.pathname.replace(/^\//, '')}`)

  const labelCounts = new Array(10).fill(0)
  for (const l of train.labels) labelCounts[l]++
  console.log('Training label distribution:', labelCounts)
}

main()
