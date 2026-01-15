let hsluv = require('./hsluv.js')
const hsl = new hsluv.Hsluv

class TensorView{

  constructor(dims, mag){
    var canvas = document.createElement('canvas')
    this.width = canvas.width = dims[0] * mag[0] 
    this.height = canvas.height = dims[1] * mag[1] 
    var ctx = canvas.getContext('2d')
    this.canvas = canvas
    this.ctx = ctx
    this.dims = dims
    this.mag = mag
  }

  clear(){
    this.ctx.clearRect(0,0,this.mag[0]*this.dims[0], this.mag[1]*this.dims[1])
  }

  draw(tensor, tile=[1,1]){
    let data = tensor.dataSync()
    let img = this.ctx.createImageData(this.width, this.height)
    //console.log(data)
    data.forEach((e,i) =>{
      let rgb = [Math.floor(127*e), Math.min(255, Math.floor(255*e)), Math.max(127, Math.floor(127*e))]//*255]
      img.data[i*4] = rgb[0]//Math.floor(e*255)
      img.data[i*4+1] = rgb[1]//Math.floor(e*255)
      img.data[i*4+2] = rgb[2]//Math.floor(e*255)
      img.data[i*4+3] = 255 //Math.floor(e*255)
    })

    this.ctx.putImageData(img,0,0,0,0,this.width, this.height)
    
  }

  append(parent){
    parent.appendChild(this.canvas)
  }

}
module.exports = TensorView
