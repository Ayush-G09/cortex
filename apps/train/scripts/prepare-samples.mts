/**
 * Picks a small, fixed handful of real MNIST test digits (from the same dataset the model was evaluated against,
 * never seen during training) and saves them for the browser demo's "try a sample" feature — letting a visitor
 * without a mouse/drawing steady enough to matter still see the network classify real, unmodified handwriting
 * correctly, rather than only ever seeing their own rough sketches.
 */
import { readFileSync, writeFileSync } from 'node:fs'

interface RawData {
  testCount: number
  testImages: string
  testLabels: number[]
}

const PER_DIGIT = 3

function main() {
  const raw: RawData = JSON.parse(readFileSync(new URL('../data/mnist.json', import.meta.url), 'utf-8'))
  const bytes = Buffer.from(raw.testImages, 'base64')

  const picked: { label: number; pixels: number[] }[] = []
  const countPerDigit = new Array(10).fill(0)
  for (let i = 0; i < raw.testCount && picked.length < PER_DIGIT * 10; i++) {
    const label = raw.testLabels[i]
    if (countPerDigit[label] >= PER_DIGIT) continue
    countPerDigit[label]++
    picked.push({ label, pixels: Array.from(bytes.subarray(i * 784, i * 784 + 784)) })
  }
  picked.sort((a, b) => a.label - b.label)

  const outPath = new URL('../../web/public/samples.json', import.meta.url)
  writeFileSync(outPath, JSON.stringify(picked))
  console.log(`Wrote ${picked.length} sample digits to ${outPath.pathname.replace(/^\//, '')}`)
}

main()
