const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const jpg = require('jpeg-js')
const IMAGE_SIZE = 100*100*3

function shuffle(array) {
  var currentIndex = array.length, temporaryValue, randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

var loadData = () => {
  return new Promise((resolve, reject) => {
    let allImages = []
    fs.readdir(__dirname + '\\training_material', (err, items) => {
      if(err) return reject(err)
      var todos = []
      items.forEach(item => {
        todos.push(new Promise((resolve, reject) => {
          fs.readdir(__dirname + '\\training_material\\'+item, (errT, itemsT) => {
            if(errT) return reject(errT)
            itemsT.forEach(itemT => {
              allImages.push({label: item, loc: __dirname + '\\training_material\\'+item+'\\'+itemT})
              resolve()
            })
          })
        }))
      })
      Promise.all(todos).then(() => {
        resolve(allImages)
      })
    })
  })
}

module.exports = () => {
  return new Promise((resolve, reject) => {
    loadData().then(origImages => {
      images = shuffle(origImages)
      console.log('loaded with '+images.length+' training images, splitting to two')
      let base = images.slice(0,Math.floor(images.length/2))
      let training = images.slice(Math.floor(images.length/2))
      var returnObject = {
        origImages: origImages,
        images: base,
        trainImages: training,
        taken: 0,
        trainTaken: 0,
        loaded: {},
        readImage(path) {
          if(!this.loaded || !this.loaded[path.split('\\').slice(-1)[0].split('.')[0]])
          this.loaded[path.split('\\').slice(-1)[0].split('.')[0]] = fs.readFileSync(path)
          return jpg.decode(this.loaded[path.split('\\').slice(-1)[0].split('.')[0]], true)
        },
        imageToByteArray(image) {
          const pixels = image.data
          const numPixels = image.width * image.height;
          const values = new Int32Array(numPixels * 3);

          for (let i = 0; i < numPixels; i++) {
            for (let channel = 0; channel < 3; ++channel) {
              values[i * 3 + channel] = pixels[i * 4 + channel];
            }
          }
          return values
        },
        nextBatch(batchSize) {
          const batchImagesArray = new Int32Array(batchSize * IMAGE_SIZE)
          const batchLabelsArray = new Uint8Array(batchSize*6)
          origLabels = []
          origImgs = []
          for(let i = 0;i < batchSize;i++) {
            var image = this.imageToByteArray(this.readImage(this.images[this.taken].loc))
            var label = this.images[this.taken].label-1+1
            origLabels.push(label)
            origImgs.push(this.images[this.taken])
            batchImagesArray.set(image, i*IMAGE_SIZE)
            batchLabelsArray[i*6+label] = 1
            this.taken++
            if(this.taken >= this.images.length) {
              this.taken = 0
              this.origImages = shuffle(this.origImages)
              this.images = this.origImages.slice(0,Math.floor(this.origImages.length/2))
              this.images = shuffle(this.images)
            }
          }
          let xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE])
          let labels = tf.tensor2d(batchLabelsArray, [batchSize, 6])
          return {xs, labels, origLabels, origImgs}
        },
        nextTrainBatch(batchSize) {
          const batchImagesArray = new Int32Array(batchSize * IMAGE_SIZE)
          const batchLabelsArray = new Uint8Array(batchSize*6)
          origLabels = []
          origImgs = []
          for(let i = 0;i < batchSize;i++) {
            var image = this.imageToByteArray(this.readImage(this.trainImages[this.trainTaken].loc))
            var label = this.trainImages[this.trainTaken].label-1+1
            origLabels.push(label)
            origImgs.push(this.images[this.taken])
            batchImagesArray.set(image, i*IMAGE_SIZE)
            batchLabelsArray[i*6+label] = 1
            this.trainTaken++
            if(this.trainTaken >= this.trainImages.length) {
              this.trainTaken = 0
              this.origImages = shuffle(this.origImages)
              this.trainImages = this.origImages.slice(Math.floor(this.origImages.length/2))
              this.trainImages = shuffle(this.trainImages)
            }
          }
          let xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE])
          let labels = tf.tensor2d(batchLabelsArray, [batchSize, 6])
          return {xs, labels, origLabels, origImgs}
        }
      }

      resolve(returnObject)
    })
  })
}