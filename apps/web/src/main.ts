import { Rng, Tensor } from '@cortex/tensor'
import { MLP } from '@cortex/nn'

interface ModelData {
  config: { hiddenSize: number; batchSize: number; epochs: number; lr: number; seed: number }
  untrainedAccuracy: number
  baselineAccuracy: number
  history: { epoch: number; loss: number; testAccuracy: number }[]
  shape: number[]
  weights: number[][]
}

const loadState = document.getElementById('loadState')!
const canvas = document.getElementById('drawCanvas') as HTMLCanvasElement
const clearBtn = document.getElementById('clearBtn') as HTMLButtonElement
const predDigit = document.getElementById('predDigit')!
const predConf = document.getElementById('predConf')!
const barsEl = document.getElementById('bars')!
const chartSvg = document.getElementById('chart') as unknown as SVGSVGElement
const proofStats = document.getElementById('proofStats')!
const sampleStrip = document.getElementById('sampleStrip')!

const ctx = canvas.getContext('2d')!
ctx.fillStyle = '#000'
ctx.fillRect(0, 0, canvas.width, canvas.height)
ctx.lineCap = 'round'
ctx.lineJoin = 'round'
ctx.strokeStyle = '#fff'
ctx.lineWidth = 18

let mlp: MLP | null = null
let hasDrawing = false

fetch('/model.json')
  .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json() })
  .then((data: ModelData) => {
    // the random init here is immediately discarded by loadWeights — any seed does, since the values never get used
    mlp = new MLP(data.shape, new Rng(0))
    mlp.loadWeights(data.weights)
    loadState.textContent = `model loaded — ${(data.history.at(-1)!.testAccuracy * 100).toFixed(1)}% test accuracy`
    drawChart(data)
    renderProofStats(data)
  })
  .catch((err) => {
    loadState.textContent = `failed to load model.json: ${err.message} — run "npm run train -w apps/train" first`
  })

interface Sample { label: number; pixels: number[] }

fetch('/samples.json')
  .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json() })
  .then((samples: Sample[]) => {
    sampleStrip.innerHTML = ''
    for (const sample of samples) {
      const btn = document.createElement('button')
      btn.className = 'sample-thumb'
      btn.title = `real test digit: ${sample.label}`
      const thumb = document.createElement('canvas')
      thumb.width = 28
      thumb.height = 28
      const tctx = thumb.getContext('2d')!
      const imgData = tctx.createImageData(28, 28)
      for (let i = 0; i < 784; i++) {
        const v = sample.pixels[i]
        imgData.data[i * 4] = v
        imgData.data[i * 4 + 1] = v
        imgData.data[i * 4 + 2] = v
        imgData.data[i * 4 + 3] = 255
      }
      tctx.putImageData(imgData, 0, 0)
      btn.appendChild(thumb)
      btn.addEventListener('click', () => loadSampleIntoCanvas(sample))
      sampleStrip.appendChild(btn)
    }
  })
  .catch(() => { sampleStrip.innerHTML = '<span class="fine">no sample digits available</span>' })

function loadSampleIntoCanvas(sample: Sample) {
  const off = document.createElement('canvas')
  off.width = 28
  off.height = 28
  const octx = off.getContext('2d')!
  const imgData = octx.createImageData(28, 28)
  for (let i = 0; i < 784; i++) {
    const v = sample.pixels[i]
    imgData.data[i * 4] = v
    imgData.data[i * 4 + 1] = v
    imgData.data[i * 4 + 2] = v
    imgData.data[i * 4 + 3] = 255
  }
  octx.putImageData(imgData, 0, 0)

  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
  hasDrawing = true
  classifyPixels(sample.pixels.map((v) => v / 255))
}

// ---- drawing -----------------------------------------------------------------------------------------------
let drawing = false
let lastX = 0
let lastY = 0

function pos(e: PointerEvent): [number, number] {
  const rect = canvas.getBoundingClientRect()
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height
  return [x, y]
}

canvas.addEventListener('pointerdown', (e) => {
  drawing = true
  hasDrawing = true
  ;[lastX, lastY] = pos(e)
  ctx.beginPath()
  ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2)
  ctx.fillStyle = '#fff'
  ctx.fill()
})
canvas.addEventListener('pointermove', (e) => {
  if (!drawing) return
  const [x, y] = pos(e)
  ctx.beginPath()
  ctx.moveTo(lastX, lastY)
  ctx.lineTo(x, y)
  ctx.stroke()
  ;[lastX, lastY] = [x, y]
  scheduleClassify()
})
window.addEventListener('pointerup', () => {
  if (drawing) scheduleClassify()
  drawing = false
})

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  hasDrawing = false
  predDigit.textContent = '–'
  predConf.textContent = '–'
  barsEl.innerHTML = ''
})

let classifyScheduled = false
function scheduleClassify() {
  if (classifyScheduled) return
  classifyScheduled = true
  requestAnimationFrame(() => {
    classifyScheduled = false
    classify()
  })
}

// ---- inference ----------------------------------------------------------------------------------------------
/** MNIST's actual images are all size-normalized: each digit's own bounding box is scaled to fit a 20x20 region
 * and centered inside the full 28x28 frame — so a "1" always fills a predictable fraction of the image, not a
 * fraction that depends on how zoomed-in the person's stylus happened to be. Naively downsampling the whole
 * 280x280 drawing canvas straight to 28x28 skips all of that: a digit drawn small, or off to one side (extremely
 * easy to do by accident in a mouse-drawing box, and NOT something the training data ever looked like) comes out
 * tiny and off-center in the 28x28 input, which is far enough outside the training distribution to misclassify
 * even an unambiguous shape — this was confirmed directly: a plain vertical bar, about as clean a "1" as exists,
 * was classified as "5" before this fix, purely because it only filled a small corner of the canvas. Finding the
 * actual drawn content's bounding box and re-centering/rescaling it to match MNIST's convention is the fix. */
function downsampleTo28x28(): Float64Array {
  const full = ctx.getImageData(0, 0, canvas.width, canvas.height).data
  let minX = canvas.width, maxX = 0, minY = canvas.height, maxY = 0
  const THRESHOLD = 24 // ignore near-black background noise/anti-aliasing fringe
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      if (full[(y * canvas.width + x) * 4] > THRESHOLD) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }
  if (maxX < minX || maxY < minY) return new Float64Array(784) // nothing drawn

  const bboxW = maxX - minX + 1
  const bboxH = maxY - minY + 1
  const PAD = Math.max(bboxW, bboxH) * 0.2 // a little breathing room, same spirit as MNIST's 20-of-28 convention
  const cropX = Math.max(0, minX - PAD)
  const cropY = Math.max(0, minY - PAD)
  const cropW = Math.min(canvas.width - cropX, bboxW + PAD * 2)
  const cropH = Math.min(canvas.height - cropY, bboxH + PAD * 2)
  const side = Math.min(canvas.width, canvas.height, Math.max(cropW, cropH)) // square crop, centered on the digit
  const srcX = Math.min(Math.max(0, cropX - (side - cropW) / 2), canvas.width - side)
  const srcY = Math.min(Math.max(0, cropY - (side - cropH) / 2), canvas.height - side)

  const off = document.createElement('canvas')
  off.width = 28
  off.height = 28
  const octx = off.getContext('2d')!
  octx.imageSmoothingEnabled = true
  octx.filter = 'blur(1px)' // softens hard mouse-drawn edges toward the anti-aliased strokes the network trained on
  octx.drawImage(canvas, srcX, srcY, side, side, 0, 0, 28, 28)
  const img = octx.getImageData(0, 0, 28, 28).data
  const out = new Float64Array(784)
  for (let i = 0; i < 784; i++) out[i] = img[i * 4] / 255 // canvas is drawn white-on-black; red channel == intensity
  return out
}

function classify() {
  if (!mlp || !hasDrawing) return
  classifyPixels(downsampleTo28x28())
}

/** Runs inference on an exact 784-value pixel vector and renders the result — used both for freehand drawing
 * (where the vector comes from downsampling the canvas) and for bundled sample digits (where the vector is the
 * ORIGINAL dataset pixels, not a re-rasterized, re-blurred copy: routing a sample through the drawing canvas
 * first measurably degrades it enough to flip some predictions, which defeats the point of a "known good" demo
 * digit). */
function classifyPixels(pixels: Float64Array | number[]): void {
  if (!mlp) return
  const x = new Tensor([1, 784], Float64Array.from(pixels))
  const probs = mlp.forward(x).softmax().data
  const arr = Array.from(probs)
  const best = arr.indexOf(Math.max(...arr))

  predDigit.textContent = String(best)
  predConf.textContent = `${(arr[best] * 100).toFixed(1)}%`

  const ranked = arr.map((p, digit) => ({ p, digit })).sort((a, b) => b.p - a.p)
  barsEl.innerHTML = ranked.map(({ p, digit }) => `
    <div class="bar-row ${digit === best ? 'top' : ''}">
      <span>${digit}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <span>${(p * 100).toFixed(1)}%</span>
    </div>
  `).join('')
}

// ---- training proof chart + stats ----------------------------------------------------------------------------
function drawChart(data: ModelData) {
  const w = 560
  const h = 150
  const pad = 4
  const n = data.history.length
  const maxLoss = Math.max(...data.history.map((g) => g.loss), 1e-6)
  const x = (i: number) => pad + (i / Math.max(1, n - 1)) * (w - pad * 2)
  const yAcc = (v: number) => h - pad - v * (h - pad * 2)
  const yLoss = (v: number) => h - pad - (v / maxLoss) * (h - pad * 2)

  const accPath = data.history.map((g, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${yAcc(g.testAccuracy).toFixed(1)}`).join(' ')
  const lossPath = data.history.map((g, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${yLoss(g.loss).toFixed(1)}`).join(' ')

  chartSvg.innerHTML = `
    <path d="${lossPath}" fill="none" stroke="#8891ac" stroke-width="1.5" />
    <path d="${accPath}" fill="none" stroke="#a78bfa" stroke-width="2.25" />
  `
}

function renderProofStats(data: ModelData) {
  const final = data.history.at(-1)!.testAccuracy
  proofStats.innerHTML = `
    <div>Final accuracy<b class="good">${(final * 100).toFixed(1)}%</b></div>
    <div>Untrained net<b class="bad">${(data.untrainedAccuracy * 100).toFixed(1)}%</b></div>
    <div>Majority baseline<b class="bad">${(data.baselineAccuracy * 100).toFixed(1)}%</b></div>
  `
}
