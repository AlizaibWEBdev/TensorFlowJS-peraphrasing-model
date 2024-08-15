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
    if (!(token in vocab)) {
      vocab[token] = nextId++;
    }
  });
};

// Function to convert text to token IDs
const textToNumerical = (text) => {
  const tokens = tokenizer.tokenize(text.toLowerCase());
  return tokens.map(token => vocab[token] || 0); // Default to 0 if token not in vocab
};

// Process the CSV file
const processCsvFile = (inputFilePath, outputFilePath) => {
  const results = [];
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
      // Build vocabulary from text
      buildVocabulary(text);
      buildVocabulary(paraphrase);
      
      // Convert text and paraphrase to token IDs
      const numericalData = {
        Text: textToNumerical(text),
        Paraphrase: textToNumerical(paraphrase),
      };
      results.push(numericalData);
    }
  });

  rl.on('close', () => {
    fs.writeFileSync(outputFilePath, JSON.stringify({ data: results, vocab }, null, 2), 'utf8');
    console.log('CSV file successfully converted to token IDs!');
  });
};

// Specify input and output file paths
const inputFilePath = 'data.csv'; // Path to your CSV file
const outputFilePath = 'data-token-ids.json'; // Path to save the output JSON file

processCsvFile(inputFilePath, outputFilePath);
