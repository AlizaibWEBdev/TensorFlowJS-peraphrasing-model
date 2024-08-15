const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

// Load and preprocess data
const loadDataInChunks = (filePath, chunkSize) => {
  try {
    const rawData = fs.readFileSync(filePath);
    const { data, vocab } = JSON.parse(rawData);
    const numChunks = Math.ceil(data.length / chunkSize);
    const maxLength = getMaxLength(data);

    let chunkIndex = 0;

    return {
      getNextChunk: () => {
        if (chunkIndex >= numChunks) {
          return null;
        }

        const chunkData = data.slice(chunkIndex * chunkSize, (chunkIndex + 1) * chunkSize);
        chunkIndex++;

        const textData = chunkData.map(d => padArray(d.Text, maxLength));
        const paraphraseData = chunkData.map(d => padArray(d.Paraphrase, maxLength));

        const textTensor = tf.tensor2d(textData, [textData.length, maxLength]);

        // Ensure one-hot encoding produces consistent sizes
        const oneHotEncodedParaphrase = paraphraseData.map(seq => oneHotEncode(seq, maxLength, vocab.length));

        // Debugging sizes
        const expectedSize = paraphraseData.length * maxLength * vocab.length;
        const actualSize = oneHotEncodedParaphrase.reduce((sum, seq) => sum + seq.flat().length, 0);

        if (expectedSize !== actualSize) {
          console.error(`Size mismatch: Expected ${expectedSize}, but got ${actualSize}`);
        }

        const paraphraseTensor = tf.tensor3d(oneHotEncodedParaphrase, [paraphraseData.length, maxLength, vocab.length]);

        return {
          textTensor,
          paraphraseTensor,
        };
      },
      vocab,
      numClasses: vocab.length,
      maxLength,
    };
  } catch (error) {
    console.error('Error loading data:', error);
    return null;
  }
};

const getMaxLength = (data) => Math.max(...data.map(d => Math.max(d.Text.length, d.Paraphrase.length)));

const padArray = (arr, length) => arr.concat(new Array(length - arr.length).fill(0)).slice(0, length);

const oneHotEncode = (seq, maxLength, numClasses) => {
  const encoded = Array.from({ length: maxLength }, () => Array(numClasses).fill(0));
  seq.forEach((token, index) => {
    if (index < maxLength && token < numClasses) {  // Ensure token index is within bounds
      encoded[index][token] = 1;
    }
  });
  return encoded;
};

// Build and compile the model
const buildOptimizedModel = (inputShape, numClasses) => {
  const model = tf.sequential();

  model.add(tf.layers.embedding({ inputDim: numClasses, outputDim: 32, inputLength: inputShape }));
  model.add(tf.layers.lstm({ units: 32, returnSequences: true }));
  model.add(tf.layers.timeDistributed({ layer: tf.layers.dense({ units: numClasses, activation: 'softmax' }) }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// Train the model
const trainModel = async () => {
  try {
    const chunkSize = 64; // Adjust as needed
    const dataLoader = loadDataInChunks('data.json', chunkSize);

    if (!dataLoader) {
      throw new Error('Failed to load data');
    }

    const model = buildOptimizedModel(dataLoader.maxLength, dataLoader.numClasses);

    let chunk;
    while ((chunk = dataLoader.getNextChunk())) {
      try {
        const { textTensor, paraphraseTensor } = chunk;

        await model.fit(textTensor, paraphraseTensor, {
          epochs: 1,
          batchSize: chunkSize,
          validationSplit: 0.1,
        });

        tf.dispose([textTensor, paraphraseTensor]); // Clean up memory
      } catch (error) {
        console.error('Error processing chunk:', error);
      }
    }

    await model.save('file://./final-model');
    console.log('Model trained and saved!');
  } catch (error) {
    console.error('Error training model:', error);
  }
};

trainModel();
