const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Load the model
const modelPath =  'final-model/model.json'
let model;

const loadModel = async () => {
  model = await tf.loadLayersModel(`file://${modelPath}`);
  console.log('Model loaded successfully.');
};

// Convert text to numerical IDs based on vocab
const vocabPath = path.join(__dirname, 'vocab.json'); // Update with your vocab path
const vocab = JSON.parse(fs.readFileSync(vocabPath, 'utf8'));

const textToNumerical = (text) => {
  const tokens = text.toLowerCase().split(/\s+/); // Simple tokenization
  return tokens.map(token => vocab[token] || 0); // Map tokens to IDs, default to 0 if not found
};

const maxInputLength = 59; // Replace with your model's expected input length

// Predict function
const predict = async (text) => {
  if (!model) {
    console.error('Model is not loaded.');
    return;
  }

  const inputIds = textToNumerical(text);
  const inputTensor = tf.tensor([inputIds], [1, maxInputLength], 'int32');

  try {
    const prediction = model.predict(inputTensor);
    processPrediction(prediction);
  } catch (error) {
    console.error('Error during prediction:', error);
  }
};

// Process prediction based on the output type
const processPrediction = (prediction) => {
  const outputArray = prediction.arraySync(); // Convert tensor to array

  // Example: If it's classification probabilities
  // Uncomment if your model's output is classification probabilities
  /*
  const probabilities = outputArray[0];
  const predictedClass = tf.argMax(probabilities).dataSync()[0];
  console.log('Predicted class:', predictedClass);
  */

  // Example: If it's embeddings or feature vectors
  console.log('Predicted embeddings:', outputArray[0]);

  // Further processing can be done here depending on your use case
};

// Example usage
const main = async () => {
  await loadModel();
  const testText = 'I am feeling good today'; // Replace with your test text
  await predict(testText);
};

main().catch(console.error);
