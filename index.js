const tf = require('@tensorflow/tfjs-node')
console.log('backend in use:', tf.getBackend())
function deltaToString(start, end) {
  let isSec = end - start
    var ms = isSec,
        lm = ~(4 * !!isSec),  /* limit fraction */
        fmt = new Date(ms).toISOString().slice(11, lm);

    if (ms >= 8.64e7) {  /* >= 24 hours */
        var parts = fmt.split(/:(?=\d{2}:)/);
        parts[0] -= -24 * (ms / 8.64e7 | 0);
        return parts.join(':');
    }

    return fmt;
}

const data = require('./data')
data().then(async dataObject => {
  //mallin määritys
  const model = tf.sequential() //luodaan verkko
  /*let uniqueName = "layer_id1"
  model.add(tf.layers.conv2d({
    inputShape: [100, 100, 3], //kuvat on 100x100 RGB kuvia
    kernelSize: 2, //loput ovat oletuksia ohjeista
    filters: 16,
    strides: 4,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }))
  model.add(tf.layers.conv2d({ kernelSize: 2, filters: 32, strides: 8, activation: 'relu', kernelInitializer: 'varianceScaling' }));                                       
  model.add(tf.layers.conv2d({ kernelSize: 2, filters: 64, strides: 4, activation: 'relu', kernelInitializer: 'varianceScaling' }));                                       
                                                                                                                                                                          
  model.add(tf.layers.flatten());                                                                                                                                         
  model.add(tf.layers.dense({ units: 30, kernelInitializer: 'varianceScaling', activation: 'softmax' }));                                                                 
  model.add(tf.layers.dense({ units: 6, activation: 'linear' }) );*/

  let uniqueName = 'layer_id3'
  model.add(tf.layers.conv2d({
    inputShape: [100, 100, 3], //kuvat on 100x100 RGB kuvia
    kernelSize: 2,
    filters: 32,
    strides: 2,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }))
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))                          
  model.add(tf.layers.conv2d({ kernelSize: 2, filters: 32, strides: 8, activation: 'relu', kernelInitializer: 'varianceScaling' }));                                       
  model.add(tf.layers.conv2d({ kernelSize: 2, filters: 64, strides: 4, activation: 'relu', kernelInitializer: 'varianceScaling' }));                                                                                                                                                                               
  model.add(tf.layers.flatten());                                                                                                                                         
  model.add(tf.layers.dense({ units: 32, kernelInitializer: 'varianceScaling', activation: 'softmax' }));
  model.add(tf.layers.dense({ units: 6, activation: 'linear' }) );       


  //mallin koulutus, stokastinen gradienttilaskeuma
  const LEARNING_RATE = 0.02
  const optimizer = tf.train.sgd(LEARNING_RATE)

  //häviöfunktiona käytetään tensorflown omaa
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  })
  //KOULUTUS
  //kuinka suuri satunnainen kasa kuvia otetaan
  const BATCH_SIZE = 50
  const TEST_BATCH_SIZE = 500
  const TEST_ITERATION_FREQUENCY = 5
  var bestAccuracy = 0;
  //kuinka monta kertaa
  const TRAIN_BATCHES = 5000
  console.log('data loaded :)')
  var startTime = Date.now()
  let i = 0
  while (bestAccuracy < 0.99 && i < TRAIN_BATCHES) {
    const batch = dataObject.nextBatch(BATCH_SIZE);
   
    let testBatch;
    let validationData;
    // Every few batches test the accuracy of the mode.
    if (i % TEST_ITERATION_FREQUENCY === 0) {
      console.log('new batch of test data loaded')
      testBatch = dataObject.nextTrainBatch(TEST_BATCH_SIZE);
      validationData = [
        testBatch.xs.reshape([TEST_BATCH_SIZE, 100, 100, 3]), testBatch.labels
      ];
    }
   
    // The entire dataset doesn't fit into memory so we call fit repeatedly
    // with batches.
    const history = await model.fit(
        batch.xs.reshape([BATCH_SIZE, 100, 100, 3]),
        batch.labels,
        {
          batchSize: BATCH_SIZE,
          validationData,
          verbose: 0,
          yieldEvery: 'never',
          epochs: 500,
          callbacks :  {
            onBatchEnd: async (batch, logs) => {
            },
           onEpochEnd: async (epoch, logs) => {
             /*process.stdout.clearLine()
             process.stdout.cursorTo(0)
             process.stdout.write(deltaToString(startTime, Date.now())+" - "+epoch+'/500','epoch ended', 'loss:', logs.loss, 'accuracy:', logs.acc)    */       
             console.log(deltaToString(startTime, Date.now())+" - total accuracy: "+bestAccuracy+" - trainiing batch: "+i+" - "+epoch+'/500','epoch ended', 'loss:', logs.loss, 'accuracy:', logs.acc)                     
           }   
                                                                                
        }  // end all callbacks
        }
      );
  
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];
    
    if ((i % 5 === 0)){
      console.log('example')
      tf.tidy(() => {
        let testAmount = 30
        let examples = dataObject.nextBatch(testAmount)
        let output = model.predict(examples.xs.reshape([testAmount, 100, 100, 3]))
        var labels = Array.from(examples.labels.argMax(1).dataSync())
        var predictions = Array.from(output.argMax(1).dataSync())
        console.log(labels, 'given labels')
        console.log(predictions, 'net predictions')
        console.log(examples.origLabels, 'original labels')
      })
    } 


    console.log('loss: ', loss, typeof loss, loss)
    console.log('accuracy', accuracy, typeof accuracy)
    bestAccuracy = accuracy > bestAccuracy ? accuracy : bestAccuracy
    console.log('best accuracy', bestAccuracy)
    //save model on every 5th batch and by the end
    console.log('--------------------end of loop '+i+'----------------------')
    i++
  }
  try {
    await model.save('file://'+__dirname+'/models/'+uniqueName+'-'+new Date().toISOString().split('T')[0]+'-complete-'+bestAccuracy)
  } catch(e) {
    console.log(e)
  }
  console.log('training done, doing debug: ')
  tf.tidy(() => {
    let examples = dataObject.nextBatch(5)
    let output = model.predict(examples.xs.reshape([5, 100, 100, 3]))
    var labels = Array.from(examples.labels.argMax(1).dataSync())
    var predictions = Array.from(output.argMax(1).dataSync())
    console.log('actual labels', labels)
    console.log('predictions', predictions)
    console.log('training data num', dataObject.images.length)
    console.log('testing data num', dataObject.trainImages.length)
  })
}).catch(console.log)

