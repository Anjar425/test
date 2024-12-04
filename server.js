require('dotenv').config();

const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');
const openai = require('openai');

process.env.TF_ENABLE_ONEDNN_OPTS = '0';

// Initialize OpenAI API client
const openaiClient = new openai.OpenAI({
    apiKey: process.env.OPENAI_API_KEY,  // Set API key from environment variables
});

const app = express(); // Initialize the app here
app.use(cors()); // Use CORS
app.use(bodyParser.json());

const corsOptions = {
    origin: 'http://localhost:3000', // URL of frontend
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type'],
};

app.use(cors(corsOptions)); // CORS options applied after app initialization

app.post('/predict', (req, res) => {
    const { text } = req.body;
    if (!text) {
        return res.status(400).json({ error: 'Text is required' });
    }

    const pythonProcess = spawn('python', ['predict.py', text]);

    let result = '';
    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
        console.log('Python script result:', result);
        if (code !== 0) {
            return res.status(500).json({ error: 'Error in prediction script' });
        }
        try {
            const predictions = JSON.parse(result);
            res.json(predictions);
        } catch (error) {
            console.error('Failed to parse Python response:', error);
            res.status(500).json({ error: 'Failed to parse Python response' });
        }
    });
});

// Endpoint to fetch suggestions from GPT based on emotion
app.get('/suggestions', async (req, res) => {
    const { emotion, text } = req.query;
    console.log(`Received emotion: ${emotion}, with text: ${text}`);

    if (!emotion || !text) {
        return res.status(400).json({ error: 'Emotion and text are required' });
    }

    try {
        const prompt = `Based on the predicted emotion: ${emotion}, and the issue: ${text}, provide suggestions to improve mental well-being.`;

        // Send request to OpenAI's GPT model
        const gptResponse = await openaiClient.chat.completions.create({
            model: 'gpt-4',  // Use GPT-4 for best results
            messages: [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: prompt }
            ],
            max_tokens: 250,
            temperature: 0.85,
        });

        const suggestion = gptResponse.choices[0].message.content.trim();

        // Convert the string into an array of suggestions (for the frontend)
        const suggestionsArray = suggestion.split('\n').map(s => s.trim()).filter(Boolean);

        res.json({ suggestions: suggestionsArray });
    } catch (error) {
        console.error('Error fetching suggestions from GPT:', error);
        res.status(500).json({ error: 'Failed to get suggestions' });
    }
});

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
