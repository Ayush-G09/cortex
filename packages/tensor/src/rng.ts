/** Seeded PRNG (mulberry32) — deterministic weight initialization, same discipline as every other project in this
 * portfolio: a given seed always produces the same initial weights, which is what makes a training run replayable
 * and a "did it actually learn" comparison fair (two runs that started identically only differ by what training
 * itself did). */
export class Rng {
  private state: number

  constructor(seed: number) {
    this.state = seed >>> 0
  }

  next(): number {
    let t = (this.state += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  /** standard normal via Box-Muller */
  gaussian(): number {
    const u1 = Math.max(this.next(), 1e-12)
    const u2 = this.next()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
}
