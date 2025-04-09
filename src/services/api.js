// This service handles all API requests.

// =========================================================================
// == SECURITY WARNING =====================================================
// =========================================================================
// == DO NOT EXPOSE API KEYS IN FRONTEND CODE! =============================
// The `EXTERNAL_AI_API_KEY` below is hardcoded for demonstration purposes
// ONLY. In a real application, this key MUST be kept secret on a backend
// server. Your frontend should call YOUR backend, which then securely uses
// the key to call the external API (like Google Gemini). Exposing keys
// here allows anyone to steal and abuse them.
// =========================================================================
const EXTERNAL_AI_API_KEY = 'AIzaSyCWHbG0Bie-Z3j1IdqhBDln6QzbYO6MFiI'; // <-- REMOVE THIS IN PRODUCTION! USE BACKEND PROXY!

// === ENSURE THESE ARE DEFINED AT THE TOP LEVEL ===
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api'; // Your Backend Base URL
const OUR_AI_ENDPOINT = `${API_BASE_URL}/analyze`; // Endpoint on YOUR backend
const REPORT_ENDPOINT = `${API_BASE_URL}/generate-report`; // Endpoint on YOUR backend
const EXTERNAL_AI_PROXY_ENDPOINT = `${API_BASE_URL}/external-analyze-proxy`; // Currently unused due to simulation
const GEMINI_VISION_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key=${EXTERNAL_AI_API_KEY}`; // <-- KEY EXPOSED! BAD PRACTICE!
// === END CONSTANT DEFINITIONS ===

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
                const base64String = resultString.split(',')[1];
                if (base64String) {
                    resolve(base64String);
                } else {
                    reject(new Error("Failed to extract Base64 string from FileReader result."));
                }
            } else {
                reject(new Error("FileReader result was null or empty"));
            }
        };
        reader.onerror = error => reject(error);
    });
}


/**
 * Helper function to handle API requests
 */
async function apiRequest(url, method, data, headers = {}) {
    const options = {
        method,
        headers: {
            'Accept': 'application/json',
            ...headers,
        },
    };
    if (data) {
        if (data instanceof FormData) {
            options.body = data;
        } else {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        }
    }
    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            let errorData;
            try { errorData = await response.json(); } catch (e) { errorData = await response.text(); }
            console.error("API Error Response:", errorData);
            throw new Error(`API error: ${response.status} ${response.statusText} - ${JSON.stringify(errorData) || errorData}`);
        }
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            const text = await response.text();
            return text ? JSON.parse(text) : { success: true, data: null };
        } else {
            const text = await response.text();
            return text ? { success: true, data: text } : { success: true, data: null };
        }
    } catch (error) {
        console.error('Network or API request failed:', error);
        throw new Error(`Failed to execute API request to ${url}: ${error.message}`);
    }
}


/**
 * Upload and analyze an image with your custom AI (via your backend)
 * MODIFIED TO SIMULATE "NOT IMPLEMENTED" STATE
 */
export async function analyzeWithOurAI(imageData) {
    // Use the constant defined at the top level
    console.log(`Simulating call to backend: POST ${OUR_AI_ENDPOINT} - Not Implemented`);
    await new Promise(resolve => setTimeout(resolve, 500));

    return {
        success: false,
        source: 'Our Custom AI Model',
        status: 'backend_unavailable',
        message: 'The backend analysis service is not yet implemented or is currently unavailable.',
        description: 'Analysis could not be performed.',
        timestamp: new Date().toISOString()
    };
}

/**
 * Upload and analyze an image with external AI service (Gemini)
 * !!! SIMULATION REQUIRING BACKEND PROXY IN PRODUCTION !!!
 * UPDATED RESPONSE STRUCTURE FOR BETTER DISPLAY
 */
export async function analyzeWithExternalAI(imageData) {
    let base64Image;
    try { base64Image = await fileToBase64(imageData.file); }
    catch (error) { throw new Error("Could not process image file for external analysis."); }

    const prompt = "Analyze this chest X-ray concisely. State the primary finding and a brief reason.";
    const geminiPayload = { /* ... */ };

    try {
        console.warn(`SIMULATING call to Gemini via a backend proxy.`);
        await new Promise(resolve => setTimeout(resolve, 2000));

        return {
            success: true,
            source: 'External AI (Simulated Gemini)',
            keyFinding: "Possible interstitial changes",
            supportingEvidence: "Diffuse reticular patterns noted bilaterally.",
            timestamp: new Date().toISOString()
        };

    } catch (error) {
        console.error('External AI Analysis Error:', error);
        throw new Error(`External AI analysis failed: ${error.message}. Ensure the backend proxy is running or check external service status.`);
    }
}


/**
 * Generate a comprehensive report combining analyses (via your backend)
 * UPDATED TO RETURN STRUCTURED DATA FOR MODERN UI & CONCISE SUMMARY
 */
export async function generateReport(imageData, analysisResult, externalAnalysisResult) {
    const reportPayload = { /* ... */ };

    try {
        // Use the constant defined at the top level
        console.log(`Simulating call to YOUR backend for report: POST ${REPORT_ENDPOINT}`);
        await new Promise(resolve => setTimeout(resolve, 2500));

        let primarySummary = "Primary AI analysis was not available.";
        if (analysisResult?.success) {
            primarySummary = analysisResult?.description?.split('.')[0] || "Primary analysis data unavailable.";
        } else if (analysisResult?.status === 'backend_unavailable'){
            primarySummary = "Primary AI backend is unavailable.";
        }
        const externalSummary = externalAnalysisResult?.keyFinding ? `External AI noted: ${externalAnalysisResult.keyFinding}.` : "External AI analysis data unavailable.";
        const consolidatedSummary = `AI systems reviewed the image. ${primarySummary} ${externalSummary} Potential findings require professional correlation.`;

        return {
            success: true,
            source: 'Combined Analysis Report',
            sections: [
                { title: "Primary AI Summary", content: analysisResult?.success ? `${analysisResult.description} (Confidence: ${(analysisResult.confidence * 100 || 0).toFixed(0)}%)` : analysisResult?.message || "Primary analysis not performed.", type: 'text' },
                { title: "External AI Opinion", content: externalAnalysisResult ? `Finding: ${externalAnalysisResult.keyFinding || 'N/A'}. Evidence: ${externalAnalysisResult.supportingEvidence || 'N/A'}` : "External analysis not performed or failed.", type: 'key_value' },
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