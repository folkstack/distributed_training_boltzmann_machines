const $ = require('./utils.js')
const argv = require('minimist')(process.argv.slice(2))
const tf = $.tf

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
      dev: .25,
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
    var prob =tf.dot(data, this.weights)//.add(this.hbias))
    var result = prob.greaterEqual(target || $.variable({init:'randomUniform', shape: [data.shape[0], this._shape[1]]}).layer).asType('float32')
    return {result, prob}
  }
  activate(data, target){
    var prob = tf.sigmoid(tf.dot(data, this.weights.transpose()).add(this.vbias))
    var result = prob.greaterEqual(target || $.variable({init:'randomUniform', shape: [data.shape[0], this._shape[0]]}).layer).asType('float32')
    return {result, prob} 
  }
  trainp(data){
    var qprob =tf.dot(data, this.weights)
    var qresult = tf.sigmoid(qprob).greaterEqual($.variable({init:'randomUniform', shape: [data.shape[0], this._shape[1]]}).layer).asType('float32')
    var hprob = tf.dot(qresult, this.weights.transpose())
    hprob = tf.sigmoid(hprob)
    var hresult = hprob.greaterEqual($.variable({init:'randomUniform', shape: [data.shape[0], this._shape[0]]}).layer).asType('float32')
    var q2 = tf.sigmoid(tf.dot(hresult, this.weights))
    var pas = tf.dot(qresult.transpose(), data)
    var nas = tf.dot(q2.transpose(), hprob)

    var error = tf.sum(data.sub(hprob).pow($.scalar(2)))
    var grad = this.rate.mul(pas.sub(nas).transpose().div($.scalar(data.shape[0])))

    this.weights.assign(this.weights.add(grad))
    this.rate = tf.variable(this.momentum.mul(this.rate).sub(grad))

    return {result: hresult, error, cost: pas.sub(nas).mean(), rate: this.rate}
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
  var brainz = new Array(10).fill(0).map((e, i) => new Boltzman(size, size * (argv.s || 4), 1e-3,  .9, 1e-6, .1, i))
  train()
 
  function train(){
    let batches = new Array(10).fill(0).map(e => [])
    var ready = () => batches.map(e => e.length >= size/2).filter(Boolean).length === batches.length
    var go = false
    while(!ready()){
      let data = mnist.nextTrainBatch(size)
      let image = data.image.reshape([size, size]).greaterEqual($.scalar(.6785)).cast('float32')
      let d = tf.unstack(image)
      let l = tf.unstack(data.label)
      
      l.forEach((e,i) => batches[tf.argMax(e).dataSync()[0]].push(d[i]))
    }
    batches = batches.map(e => tf.stack(e.slice(0, size)))
    batches.forEach( (e, n) => {
      for(var i = 0; i < epochas; i++)
        tf.tidy(_ => {
          let res = brainz[n].trainp(e)
          res.cost.print()
          if(i==epochas-1){
            tf.unstack(res.result).slice(0,1).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
            //res.error.print()
            //if(argv.save) brainz[n].save()
            }
          })
      
    })

    var bm = new Boltzman(size, size * (argv.s || 4), 1e-2,  .9, 1e-6, .1, 10)
    bm.weights = brainz.slice(1).reduce((a, e) => a.add(e.weights), brainz[0].weights)
    batches.forEach(batch =>{
  tf.tidy(_=>{
    tf.unstack(bm.activate(bm.query(batch).result).result).slice(0, 1).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
  })
})
  }
}

