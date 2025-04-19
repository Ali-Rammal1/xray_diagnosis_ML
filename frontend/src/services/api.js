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
 */
export async function analyzeWithOurAI(imageFile) {
    const OUR_AI_ENDPOINT = 'http://localhost:5000/predict';

    try {
        if (!imageFile) throw new Error("Image file is required for analysis.");

        const formData = new FormData();
        formData.append('image', imageFile);

        const response = await fetch(OUR_AI_ENDPOINT, {
            method: 'POST',
            body: formData,
        });

        console.log(`Our AI Response Status: ${response.status}`);
        const responseText = await response.text();
        console.log("Our AI Raw Response:", responseText);

        let data;
        try {
            data = JSON.parse(responseText);
        } catch (e) {
            console.error("Our AI Response was not valid JSON:", e);
            throw new Error(`Server returned non-JSON response: ${response.status} - ${responseText.substring(0, 100)}`);
        }


        if (!response.ok) {
            const errorMessage = data?.message || `Failed to get prediction from server (${response.status}).`;
            throw new Error(errorMessage);
        }

        let confidenceLevel = 'low';
        if (typeof data.confidence === 'number') {
            if (data.confidence >= 0.8) {
                confidenceLevel = 'high';
            } else if (data.confidence >= 0.5) {
                confidenceLevel = 'medium';
            }
            console.log(`Our AI Confidence: ${data.confidence.toFixed(3)} -> Level: ${confidenceLevel}`);
        } else {
            console.warn("Confidence value missing or not a number in Our AI response:", data.confidence);
        }


        let diagnosisMessage = '';
        const diagnosis = data.prediction || 'unknown';

        if (diagnosis.toLowerCase() === 'normal') {
            if (confidenceLevel === 'high') {
                diagnosisMessage = `The primary AI analysis indicates the lungs appear normal with high confidence. No significant abnormalities detected.`;
            } else if (confidenceLevel === 'medium') {
                diagnosisMessage = `The primary AI analysis suggests a normal lung condition with medium confidence. Some subtle features may warrant professional review.`;
            } else {
                diagnosisMessage = `The primary AI analysis suggests a normal lung condition, but with low confidence. Professional interpretation is recommended.`;
            }
        } else if (diagnosis.toLowerCase() === 'pneumonia' || diagnosis.toLowerCase() === 'tuberculosis') {
            if (confidenceLevel === 'high' || confidenceLevel === 'medium') {
                diagnosisMessage = `The primary AI analysis suggests ${diagnosis} with ${confidenceLevel} confidence based on detected radiographic features.`;
            } else {
                diagnosisMessage = `The primary AI analysis detected potential signs of ${diagnosis}, but with low confidence. Professional correlation is essential.`;
            }
        } else {
            diagnosisMessage = `The primary AI analysis resulted in an '${diagnosis}' classification with ${confidenceLevel} confidence. Professional interpretation is required to understand the findings.`;
        }

        return {
            ...data,
            success: true,
            source: 'Our AI Engine',
            prediction: data.prediction || 'Unknown',
            confidence: data.confidence,
            confidenceLevel,
            diagnosisMessage,
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        console.error('Error calling Our AI:', error);
        return {
            success: false,
            source: 'Our AI Engine',
            status: 'backend_unavailable',
            message: error.message || 'Failed to connect or get a valid response from Our AI.',
            description: 'Analysis could not be performed using Our AI.',
            timestamp: new Date().toISOString()
        };
    }
}


/**
 * Upload and analyze an image with external AI service (Gemini) - FOR OPINION ONLY
 */
export async function analyzeWithExternalAI(imageData) {
    let base64Image;
    if (!imageData || !imageData.file) {
        console.error("External AI: imageData or imageData.file is missing.");
        return { success: false, source: 'External AI (Input Error)', message: 'No image file provided for external analysis.' };
    }

    try {
        base64Image = await fileToBase64(imageData.file);
        console.log("Image successfully converted to Base64 for external AI.");
    } catch (error) {
        console.error("External AI: Failed to convert image to Base64:", error);
        return { success: false, source: 'External AI (Processing Error)', message: `Could not process image file: ${error.message}` };
    }

    const prompt = "Analyze the provided chest X-ray concisely.State the primary finding and a brief reason.If the image is not a chest X-ray, respond with: " +
        "'This appears to be an image of [object type], not a chest X-ray.Analysis is specific to chest X-rays.' Determine if the chest X-ray appears normal, shows signs" +
        " of pneumonia, tuberculosis, or is inconclusive.Prioritize distinguishing between pneumonia and tuberculosis by carefully evaluating radiographic features.Choose " +
        "only one most likely diagnosis.If unsure, state that further analysis is required.Analyze key radiographic markers: for pneumonia (e.g., lower lobe consolidations, air " +
        "bronchograms), for TB (e.g., upper lobe cavities, nodules, lymphadenopathy).Compare plausible options internally, then commit to the best match based on these findings without " +
        "outputting the internal analysis.Check for weak assumptions or bias, particularly in confusing TB with pneumonia.Synthesize a final output from three internal perspectives: (1) an expert " +
        "radiologist, (2) a data-driven ML researcher, (3) a skeptical innovator.The final output should be structured: 'According to the X-ray and initial analysis, it appears that you have [disease]due " +
        "to [specific feature].' If the lungs are normal, state: 'According to the X-ray and initial analysis, the lungs appear normal, with no significant abnormalities detected.' Avoid speculative framing or " +
        "statements like 'If you were healthy...' The response must be precise, concise, and patient-facing only. Give a short and concise answer of the disease and a short reason as to why you think so. ";

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
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048
        },
    };

    try {
        console.warn(`!!! MAKING DIRECT, INSECURE CALL TO GEMINI API FROM FRONTEND (${GEMINI_API_ENDPOINT}) !!! THIS IS FOR LOCAL TESTING ONLY. DO NOT DEPLOY! NEED BACKEND PROXY!`);
        console.log("Sending payload to Gemini (1.5 Flash) for opinion:", JSON.stringify(geminiPayload).substring(0, 200) + "...");

        const response = await apiRequest(GEMINI_API_ENDPOINT, 'POST', geminiPayload);

        if (response.candidates && response.candidates[0].finishReason === 'SAFETY') {
            const safetyFeedback = response.promptFeedback?.safetyRatings || response.candidates[0]?.safetyRatings || [];
            const blockedCategories = safetyFeedback.filter(r => r.blocked).map(r => r.category).join(', ');
            throw new Error(`Analysis blocked by content safety filters (Categories: ${blockedCategories || 'Unknown'}).`);
        }
        if (!response.candidates?.[0]?.content?.parts?.[0]?.text) {
            if (response.promptFeedback?.blockReason) {
                throw new Error(`Analysis blocked by prompt feedback: ${response.promptFeedback.blockReason}`);
            }
            console.error("Unexpected Gemini response structure for opinion:", response);
            throw new Error("Received an unexpected or incomplete response from the external AI service.");
        }

        const analysisText = response.candidates[0].content.parts[0].text;
        console.log("Received opinion text from Gemini (1.5 Flash):", analysisText);

        return {
            success: true,
            source: 'External AI (Gemini 1.5 Flash)',
            description: analysisText.trim(),
            rawResponse: response,
            timestamp: new Date().toISOString()
        };

    } catch (error) {
        console.error('External AI Analysis Request Error:', error);
        let message = error.message || 'Unknown error during external analysis.';
        if (message.startsWith('API request failed: API error:')) {
            const coreErrorMatch = message.match(/API error: \d+.*? - (.*)/);
            message = coreErrorMatch ? `External AI analysis failed: ${coreErrorMatch[1]}` : message;
        } else {
            message = `External AI analysis failed: ${message}`;
        }
        return { success: false, source: 'External AI (API Error)', message };
    }
}


/**
 * Generate a comprehensive report combining analyses (LOCAL IMPLEMENTATION)
 * Improved synthesis logic.
 */
export async function generateReport(imageData, analysisResult, externalAnalysisResult) {
    console.log("Generating report locally based on available analyses...");
    console.log("Primary AI Result:", analysisResult);
    console.log("External AI Result:", externalAnalysisResult);

    const hasPrimaryResult = analysisResult?.success;
    const primaryPrediction = hasPrimaryResult ? analysisResult.prediction.toLowerCase() : null;
    const primaryConfidence = hasPrimaryResult ? analysisResult.confidenceLevel : null;

    const hasExternalResult = externalAnalysisResult?.success;
    const externalDescription = hasExternalResult ? externalAnalysisResult.description.toLowerCase() : null;

    // --- Helper to extract core finding from external text ---
    const getExternalCoreFinding = (desc) => {
        if (!desc) return null;
        if (desc.includes('normal')) return 'normal';
        if (desc.includes('pneumonia')) return 'pneumonia';
        if (desc.includes('tuberculosis') || desc.includes('tb')) return 'tuberculosis';
        if (desc.includes('not a chest x-ray') || desc.includes('not an x-ray')) return 'invalid_image';
        return 'other'; // Indicates some finding, but not one we explicitly categorize
    };

    const externalFinding = getExternalCoreFinding(externalDescription);

    // --- Build Overall Assessment ---
    let overallAssessment = "AI analysis summary: ";
    let consultNeeded = false;

    if (hasPrimaryResult && hasExternalResult) {
        // Both AIs provided results
        if (primaryPrediction === externalFinding) {
            if (primaryPrediction === 'normal') {
                overallAssessment += `Both AI systems suggest a normal finding (Primary: ${primaryConfidence} confidence).`;
                consultNeeded = primaryConfidence === 'low' || primaryConfidence === 'medium'; // Recommend checkup even if normal but not high confidence
            } else if (primaryPrediction === 'invalid_image' || externalFinding === 'invalid_image') {
                overallAssessment += `One or both systems identified the image as likely not a chest X-ray. Analysis may be invalid.`;
                consultNeeded = false; // No medical consult needed for wrong image type
            } else { // Agreement on a disease
                overallAssessment += `Both AI systems noted findings consistent with ${analysisResult.prediction} (Primary: ${primaryConfidence} confidence).`;
                consultNeeded = true; // Always consult for disease finding
            }
        } else { // Disagreement or one is 'other'/'invalid'
            if (primaryPrediction === 'invalid_image' || externalFinding === 'invalid_image') {
                overallAssessment += `One system identified the image as likely not a chest X-ray, while the other reported ${primaryPrediction === 'invalid_image' ? externalFinding : primaryPrediction}. Re-upload a valid chest X-ray if necessary.`;
                consultNeeded = false;
            } else if (primaryPrediction && externalFinding) {
                overallAssessment += `AI analyses presented differing findings (Primary: ${analysisResult.prediction} [${primaryConfidence}]; External: suggested ${externalFinding || 'other findings'}).`;
                consultNeeded = true; // Disagreement always warrants consult
            } else { // Handle cases where one finding is null due to extraction error, but both succeeded
                overallAssessment += `Primary AI suggested ${analysisResult.prediction} (${primaryConfidence}). External AI analysis provided context (${externalFinding || 'other/unclear findings noted'}). Differing perspectives observed.`;
                consultNeeded = true;
            }
        }
    } else if (hasPrimaryResult) {
        // Only Primary AI succeeded
        overallAssessment += `Primary AI suggested ${analysisResult.prediction} (${primaryConfidence} confidence). External AI analysis was unavailable.`;
        if (primaryPrediction !== 'normal' || primaryConfidence !== 'high') {
            consultNeeded = true;
        }
    } else if (hasExternalResult) {
        // Only External AI succeeded
        overallAssessment += `Primary AI analysis unavailable. External AI analysis noted findings potentially consistent with ${externalFinding || 'abnormalities'}.`;
        if (externalFinding !== 'normal' && externalFinding !== 'invalid_image') {
            consultNeeded = true;
        } else if (externalFinding === 'invalid_image') {
            overallAssessment = `External AI identified the image as likely not a chest X-ray. Primary AI analysis was unavailable.`;
            consultNeeded = false;
        }
    } else {
        // Should not happen due to initial check, but as a fallback:
        overallAssessment = "AI analysis could not be completed by either system.";
        // We already returned success: false earlier in this case.
    }

    overallAssessment += " Professional correlation is required.";


    // --- Build Recommendation ---
    let recommendation = "This AI analysis is informational only and does not constitute medical advice. ";
    if (consultNeeded) {
        recommendation += "Consult a qualified healthcare professional promptly to discuss these findings and determine appropriate next steps.";
    } else if(externalFinding === 'invalid_image' || primaryPrediction === 'invalid_image'){
        recommendation += "Please ensure the uploaded image is a valid chest X-ray for analysis.";
    }
    else {
        recommendation += "While significant abnormalities were not confidently detected by the AI, consult your healthcare provider for any concerns or routine follow-up.";
    }

    // --- Construct Report Sections ---
    const sections = [];
    if (hasPrimaryResult || !hasExternalResult) { // Show primary unless it failed AND external succeeded
        sections.push({
            title: "Primary AI Summary",
            content: hasPrimaryResult ? analysisResult.diagnosisMessage : "Primary AI analysis was not performed or failed.",
            type: 'primary',
            diagnosis: hasPrimaryResult ? primaryPrediction : null,
            confidenceLevel: hasPrimaryResult ? primaryConfidence : null
        });
    }
    if (hasExternalResult || !hasPrimaryResult) { // Show external unless it failed AND primary succeeded
        sections.push({
            title: "External AI Opinion",
            content: hasExternalResult ? externalAnalysisResult.description : "External AI analysis was not performed or failed.",
            type: 'external'
        });
    }
    sections.push({ title: "Overall Assessment", content: overallAssessment, type: 'summary' });
    sections.push({ title: "Recommendation", content: recommendation, type: 'disclaimer' });


    return {
        success: true,
        source: 'Combined Analysis Report',
        sections: sections,
        generatedAt: new Date().toISOString(),
        reportId: "rep-" + Math.random().toString(36).substring(2, 10)
    };

}