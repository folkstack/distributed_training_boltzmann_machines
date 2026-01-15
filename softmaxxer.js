const $ = require('./utils.js')
require('@tensorflow/tfjs-node')
const argv = require('minimist')(process.argv.slice(2))
const tf = $.tf

console.log(argv)
class Boltzman{
  constructor(input_size, hidden_size, rate=1e-3, momentum=.01, decay=1e-5, dev=.01, ii){
    this.input_size = input_size
    this.hidden_size = hidden_size
    this.dev = dev
    this.l1 = $.scalar(0)
    this.decay = $.scalar(decay)
    this.momentum = $.scalar(momentum)
    this._shape = [input_size, hidden_size]
    let w = $.variable({
      shape: this._shape,
      init: 'truncatedNormal'
      //id: `boltzmanBrain${ii}-i${input_size}-h${hidden_size}`
    })
    this.weights = w.layer 
    this.rate = $.scalar(rate)
    this.hbias = $.variable({
      shape: [1, hidden_size],
      init: 'zeros'
    }).layer
    this.vbias = $.variable({
      shape: [1, input_size],
      init: 'zeros'
    }).layer
  }
  query(data, target){
    var prob =tf.matMul(data, this.weights)//.add(this.hbias))
    var result = prob.greaterEqual(target || $.variable({init:'randomUniform', shape: [data.shape[0], this._shape[1]]}).layer).asType('float32')
    return {result, prob}
  }
  activate(data, target){
    var prob = tf.sigmoid(tf.matMul(data, this.weights.transpose()).add(this.vbias))
    var result = prob.greaterEqual(target || $.variable({init:'randomUniform', shape: [data.shape[0], this._shape[0]]}).layer).asType('float32')
    return {result, prob} 
  }
  trainp(data){
    var qprob = tf.sigmoid(tf.matMul(data, this.weights))
    var qresult = qprob.mul($.scalar(6)).softmax()//greaterEqual($.variable({init:'randomUniform', min:.25, max:.75, shape: [data.shape[0], this._shape[1]]}).layer).asType('float32')
    var hprob = tf.matMul(qresult, this.weights.transpose())
    hprob = tf.sigmoid(hprob)
    var hresult = hprob.greaterEqual($.variable({init:'randomUniform', min:1, max:2, shape: [data.shape[0], this._shape[0]]}).layer).asType('float32')
    var q2 = tf.sigmoid(tf.matMul(hresult, this.weights))
    var pas = tf.matMul(qresult.transpose(), data)
    var nas = tf.matMul(q2.transpose(), hprob)

    var error = tf.sum(data.sub(hprob).pow($.scalar(2)))
    var grad = this.rate.mul(pas.sub(nas).transpose().div($.scalar(data.shape[0])))

    this.weights.assign(this.weights.add(grad))
//    this.rate = tf.variable(this.momentum.mul(this.rate).sub(grad))

    return {result: hresult, error, cost: pas.sub(nas).mean(), rate: this.rate, grad: grad.mean()}
  }
}

module.exports = Boltzman

multi()

async function multi(){

  let mnist = require('./data.js')
  var chalk = require('chalk')
  var hft = require('../audio/fft/hft')
  console.vlog = _ => console.log(_.split('').map(e => Number(e) === 0 ? chalk.black.bgBlue('0') : chalk.black.bgGreen('0')).join(''))
  await mnist.loadData()
  let size = 784
  let epochas = Number(argv.e) || 8
  let embed = 4
  var brainz = new Array(10).fill(0).map((e, i) => new Boltzman(size, size * (argv.s || 4), argv.r || 1e-3,  .9, 1e-6, .1, i))
  train()
 
  function train(){
    let batches = new Array(10).fill(0).map(e => [])
    let batchSize = argv.b || size
    var ready = () => batches.map(e => e.length >= batchSize).filter(Boolean).length === batches.length
    var go = false
    while(!ready()){
      let data = mnist.nextTrainBatch(argv.b || size)
      let image = data.image.reshape([argv.b || size, size]).cast('float32')
      let d = tf.unstack(image)
      let l = tf.unstack(data.label)
      
      l.forEach((e,i) => batches[tf.argMax(e).dataSync()[0]].push(d[i]))
    }
    batches = batches.map(e => tf.stack(e.slice(0, batchSize)))
    validate()
    batches.forEach( (e, n) => {
      for(var i = 0; i < epochas; i++)
        tf.tidy(_ => {
          let res = brainz[n].trainp(e)
          console.log('COST>>>>')
          res.cost.print()
          console.log('ERROR>>>')
          res.error.print()
          console.log('GRAD>>>')
          res.grad.print()
          if(i==epochas-1){
            let q = brainz[n].activate(brainz[n].query(e).result).result
            tf.unstack(q).slice(0,1).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
            //res.error.print()
            //if(argv.save) brainz[n].save()
            }
          })
      
    })
  //validate()
  function validate(){
    var bm = new Boltzman(size, size * (argv.s || 4), argv.r || 1e-3,  .9, 1e-6, .1, 10)
    bm.weights = brainz.slice(1).reduce((a, e) => a.add(e.weights), brainz[0].weights)
    batches.forEach(batch =>{
  tf.tidy(_=>{
    tf.unstack(bm.activate(bm.query(batch).result).result).slice(0, 8).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
  })
})}
  }
}

