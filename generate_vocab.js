const fs = require('fs');
const readline = require('readline');
const natural = require('natural');

const tokenizer = new natural.WordTokenizer();
const vocab = {}; // Vocabulary for token-to-ID mapping
let nextId = 1;

// Function to build vocabulary
const buildVocabulary = (text) => {
  const tokens = tokenizer.tokenize(text.toLowerCase());
  tokens.forEach(token => {
    if (!(token in vocab) && isNaN(token)) { // Ensure token is not numeric
      vocab[token] = nextId++;
    }
  });
};

// Process the CSV file
const processCsvFile = (inputFilePath, outputFilePath) => {
  const fileStream = fs.createReadStream(inputFilePath);

  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  let isFirstLine = true;
  rl.on('line', (line) => {
    if (isFirstLine) {
      isFirstLine = false; // Skip header
      return;
    }

    const [text, paraphrase] = line.split(',');
    if (text && paraphrase) {
      // Build vocabulary from text and paraphrase
      buildVocabulary(text);
      buildVocabulary(paraphrase);
    }
  });

  rl.on('close', () => {
    fs.writeFileSync(outputFilePath, JSON.stringify(vocab, null, 2), 'utf8');
    console.log('Vocabulary file successfully created!');
  });
};

// Specify input and output file paths
const inputFilePath = 'data.csv'; // Path to your CSV file
const outputFilePath = 'vocab.json'; // Path to save the output JSON file

processCsvFile(inputFilePath, outputFilePath);
