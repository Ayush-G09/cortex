declare module 'mnist' {
  interface Sample {
    input: number[]
    output: number[]
  }
  interface MnistSet {
    training: Sample[]
    test: Sample[]
  }
  function set(trainingCount: number, testCount: number): MnistSet
  const _default: { set: typeof set }
  export default _default
}
