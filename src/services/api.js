const EXTERNAL_AI_API_KEY = 'AIzaSyCWHbG0Bie-Z3j1IdqhBDln6QzbYO6MFiI'; // <-- INSECURE! REMOVE/PROXY IN PRODUCTION!

// === ENSURE THESE ARE DEFINED AT THE TOP LEVEL ===
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api'; // Your Backend Base URL
const OUR_AI_ENDPOINT = `${API_BASE_URL}/analyze`; // Endpoint on YOUR backend
const REPORT_ENDPOINT = `${API_BASE_URL}/generate-report`; // Endpoint on YOUR backend
// const EXTERNAL_AI_PROXY_ENDPOINT = `${API_BASE_URL}/external-analyze-proxy`; // This is what you SHOULD use

const GEMINI_API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${EXTERNAL_AI_API_KEY}`; // <-- KEY EXPOSED! BAD PRACTICE! USE gemini-1.5-flash-latest


/**
 * Helper function to convert a File object to a Base64 string.
 */
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            if (reader.result) {
                const resultString = typeof reader.result === 'string' ? reader.result : '';
                // Ensure there's a comma before splitting
                if (resultString.includes(',')) {
                    const base64String = resultString.split(',')[1];
                    if (base64String) {
                        resolve(base64String);
                    } else {
                        reject(new Error("Failed to extract Base64 string after comma."));
                    }
                } else {
                    reject(new Error("FileReader result did not contain Base64 prefix."));
                }
            } else {
                reject(new Error("FileReader result was null or empty"));
            }
        };
        reader.onerror = error => reject(new Error(`FileReader error: ${error}`));
    });
}


/**
 * Helper function to handle API requests
 */
async function apiRequest(url, method, data, headers = {}) {
    const options = {
        method,
        headers: {
            // 'Accept': 'application/json', // Accept header might not be needed or desired for all APIs
            ...headers,
        },
    };
    // Gemini API expects JSON, so ensure Content-Type is set
    if (!(data instanceof FormData)) {
        options.headers['Content-Type'] = 'application/json';
    }

    if (data) {
        if (data instanceof FormData) {
            options.body = data;
        } else {
            options.body = JSON.stringify(data);
        }
    }
    try {
        console.log(`Making API Request: ${method} ${url}`); // Log the request
        const response = await fetch(url, options);
        // Log response status
        console.log(`API Response Status: ${response.status} ${response.statusText}`);

        // Try to parse JSON regardless of ok status first to get error details
        let responseData;
        try {
            responseData = await response.json();
            console.log("API Response Body (Parsed JSON):", responseData);
        } catch (e) {
            // If JSON parsing fails, try getting text
            const textResponse = await response.text();
            console.log("API Response Body (Raw Text):", textResponse);
            // If the response wasn't ok and parsing failed, throw based on text
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText} - ${textResponse || 'No response body'}`);
            }
            // If response was ok but not JSON, return the text wrapped? Or handle differently?
            // For Gemini, we expect JSON, so failing to parse JSON on a 200 OK is unexpected.
            // Let's assume non-JSON on OK is an error case for Gemini.
            throw new Error(`API Success (Status ${response.status}) but failed to parse JSON response. Body: ${textResponse}`);
        }

        if (!response.ok) {
            // Throw using the parsed JSON error data if available
            throw new Error(`API error: ${response.status} ${response.statusText} - ${JSON.stringify(responseData) || 'Failed to parse error details'}`);
        }

        // If response is ok and JSON was parsed, return the data
        return responseData;

    } catch (error) {
        // Catch both network errors and thrown errors from response handling
        console.error(`API Request Failed: ${method} ${url}`, error);
        // Rethrow a cleaner error message for the UI to potentially catch
        throw new Error(`API request failed: ${error.message}`);
    }
}


/**
 * Upload and analyze an image with your custom AI (via your backend)
 * SIMULATION REMAINS - This part likely needs a real backend endpoint
 */
export async function analyzeWithOurAI(imageData) {
    console.log(`Simulating call to backend: POST ${OUR_AI_ENDPOINT} - Not Implemented`);
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate delay

    // Return a failure state as it's not implemented
    return {
        success: false,
        source: 'Our Custom AI Model',
        status: 'backend_unavailable',
        message: 'The backend analysis service is not yet implemented or is currently unavailable.',
        description: 'Analysis could not be performed using Our AI.', // More specific description
        timestamp: new Date().toISOString()
    };
}

/**
 * Upload and analyze an image with external AI service (Gemini)
 * USES UPDATED GEMINI 1.5 FLASH ENDPOINT
 */
export async function analyzeWithExternalAI(imageData) {
    let base64Image;
    try {
        base64Image = await fileToBase64(imageData.file);
        console.log("Image successfully converted to Base64.");
    } catch (error) {
        console.error("Failed to convert image to Base64:", error);
        throw new Error(`Could not process image file for external analysis: ${error.message}`);
    }

    // Your specific prompt for Gemini
    const prompt = "Analyze the provided chest X-ray concisely.State the primary finding and a brief reason.If the image is not a chest X-ray, respond with: " +
        "'This appears to be an image of [object type], not a chest X-ray.Analysis is specific to chest X-rays.' Determine if the chest X-ray appears normal, shows signs" +
        " of pneumonia, tuberculosis, or is inconclusive.Prioritize distinguishing between pneumonia and tuberculosis by carefully evaluating radiographic features.Choose " +
        "only one most likely diagnosis.If unsure, state that further analysis is required.Analyze key radiographic markers: for pneumonia (e.g., lower lobe consolidations, air " +
        "bronchograms), for TB (e.g., upper lobe cavities, nodules, lymphadenopathy).Compare plausible options internally, then commit to the best match based on these findings without " +
        "outputting the internal analysis.Check for weak assumptions or bias, particularly in confusing TB with pneumonia.Synthesize a final output from three internal perspectives: (1) an expert " +
        "radiologist, (2) a data-driven ML researcher, (3) a skeptical innovator.The final output should be structured: 'According to the X-ray and initial analysis, it appears that you have [disease]due " +
        "to [specific feature].' If the lungs are normal, state: 'According to the X-ray and initial analysis, the lungs appear normal, with no significant abnormalities detected.' Avoid speculative framing or " +
        "statements like 'If you were healthy...' The response must be precise, concise, and patient-facing only.If there's a 50/50 between pneumonia and TB, choose TB, but ensure that pneumonia is mentioned if it is still the diagnosis.";

    // Construct the payload for the Gemini API (structure remains the same)
    const geminiPayload = {
        "contents": [{
            "parts": [
                { "text": prompt },
                {
                    "inline_data": {
                        "mime_type": imageData.file.type,
                        "data": base64Image
                    }
                }
            ]
        }],
        // Optional: Add generation configuration if needed for 1.5 flash
        "generationConfig": {
            "temperature": 0.4, // Example: adjust creativity
            "maxOutputTokens": 2048 // Example: set max length
        },
        // Optional: Safety settings (structure is generally consistent)
        // "safetySettings": [ ... ]
    };

    try {
        // --- ACTUAL API CALL (INSECURE - FOR LOCAL TESTING ONLY) ---
        // Using the updated GEMINI_API_ENDPOINT constant
        console.warn(`!!! MAKING DIRECT, INSECURE CALL TO GEMINI API FROM FRONTEND (${GEMINI_API_ENDPOINT}) !!! THIS IS FOR LOCAL TESTING ONLY. DO NOT DEPLOY! USE A BACKEND PROXY!`);
        console.log("Sending payload to Gemini (1.5 Flash):", JSON.stringify(geminiPayload).substring(0, 200) + "...");

        const response = await apiRequest(GEMINI_API_ENDPOINT, 'POST', geminiPayload); // Use the updated endpoint

        // --- PARSE THE ACTUAL GEMINI RESPONSE ---
        // Check for safety ratings / blocks first (important!)
        if (response.candidates && response.candidates[0].finishReason === 'SAFETY') {
            console.error("Gemini response blocked due to safety settings.");
            // Extract safety rating details if available
            const safetyFeedback = response.promptFeedback?.safetyRatings || response.candidates[0]?.safetyRatings || [];
            const blockedCategories = safetyFeedback.filter(r => r.blocked).map(r => r.category).join(', ');
            throw new Error(`Analysis blocked by content safety filters (Categories: ${blockedCategories || 'Unknown'}).`);
        }
        if (!response.candidates || !response.candidates[0] || !response.candidates[0].content || !response.candidates[0].content.parts || !response.candidates[0].content.parts[0].text) {
            // Handle potential lack of response content even if API call was 'ok'
            if(response.promptFeedback?.blockReason){
                throw new Error(`Analysis blocked by prompt feedback: ${response.promptFeedback.blockReason}`);
            }
            console.error("Unexpected Gemini response structure:", response);
            throw new Error("Received an unexpected or incomplete response from the AI analysis service.");
        }

        const analysisText = response.candidates[0].content.parts[0].text;
        console.log("Received analysis text from Gemini (1.5 Flash):", analysisText);

        return {
            success: true,
            source: 'External AI (Gemini 1.5 Flash)', // Update source name
            description: analysisText.trim(), // The main analysis text from Gemini
            rawResponse: response, // Optionally include the raw response for debugging
            timestamp: new Date().toISOString()
        };
        // --- END ACTUAL API CALL ---

    } catch (error) {
        console.error('External AI Analysis Request Error:', error);
        // Check if the error message already contains useful info from apiRequest
        if (error.message.startsWith('API request failed: API error:')) {
            // Extract the core API error message if possible
            const coreErrorMatch = error.message.match(/API error: \d+.*? - (.*)/);
            const coreError = coreErrorMatch ? coreErrorMatch[1] : error.message;
            throw new Error(`External AI analysis failed: ${coreError}`);
        } else {
            // Otherwise, use the caught error's message directly
            throw new Error(`External AI analysis failed: ${error.message}`);
        }
    }
}


/**
 * Generate a comprehensive report combining analyses (via your backend)
 * SIMULATION REMAINS - This needs a real backend endpoint
 * Updated to handle potentially failed external analysis in summary.
 */
export async function generateReport(imageData, analysisResult, externalAnalysisResult) {
    // Payload for your backend (if you build one)
    const reportPayload = {
        imageFileName: imageData.name,
        ourAnalysis: analysisResult,
        externalAnalysis: externalAnalysisResult // Pass the result object
    };

    try {
        console.log(`Simulating call to YOUR backend for report: POST ${REPORT_ENDPOINT}`);
        await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate shorter delay

        // Build the report based on the success/failure of each analysis
        let primarySummary = "Primary AI analysis was not available or failed.";
        if (analysisResult?.success) {
            primarySummary = `${analysisResult.description} (Confidence: ${(analysisResult.confidence * 100 || 0).toFixed(0)}%)`;
        } else if (analysisResult?.message) {
            primarySummary = `Primary AI Status: ${analysisResult.message}`;
        }

        let externalSummary = "External AI analysis was not performed or failed.";
        if (externalAnalysisResult?.success) {
            externalSummary = externalAnalysisResult.description; // Use the description from Gemini
        } else if (externalAnalysisResult?.message) { // If external call failed gracefully
            externalSummary = `External AI Status: ${externalAnalysisResult.message}`;
        } else if (externalAnalysisResult === null) { // Explicitly check if it wasn't run
            externalSummary = "External AI analysis was not initiated.";
        }


        const consolidatedSummary = `AI systems reviewed the image "${imageData.name}". \nPrimary AI backend is currently ${analysisResult?.success ? 'operational (see details)' : 'unavailable'}. \nExternal AI's (Gemini 1.5 Flash) assessment: ${externalAnalysisResult?.success ? 'See details below.' : 'Analysis unavailable or failed.'} Potential findings require professional correlation.`;

        return {
            success: true,
            source: 'Combined Analysis Report',
            // Use a more structured format if your UI can handle it, otherwise keep text.
            // This example keeps the previous section structure.
            sections: [
                { title: "Primary AI Summary", content: primarySummary, type: 'text' },
                { title: "External AI Opinion (Gemini 1.5 Flash)", content: externalSummary, type: 'text' }, // Updated title and uses full description
                { title: "Overall Assessment", content: consolidatedSummary, type: 'summary' },
                { title: "Recommendation", content: "This AI analysis is informational only and does not constitute medical advice. Please consult a qualified healthcare professional immediately to discuss these potential findings.", type: 'disclaimer' }
            ],
            generatedAt: new Date().toISOString(),
            reportId: "rep-" + Math.random().toString(36).substring(2, 10)
        };

    } catch (error) {
        console.error('Report Generation Error:', error);
        throw new Error(`Failed to generate report: ${error.message}. Please ensure the backend service is available.`);
    }
}