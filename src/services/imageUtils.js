/**
 * Validates image files for format and size
 */
export async function validateImage(file) {
    // Check if it's an image
    if (!file.type.match('image.*')) {
        return {
            valid: false,
            error: 'Please upload an image file (JPEG, PNG, or WebP)'
        };
    }

    // Check file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        return {
            valid: false,
            error: 'Only JPEG, PNG, and WebP formats are supported'
        };
    }

    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
        return {
            valid: false,
            error: 'Image size should be less than 10MB'
        };
    }

    // Additional validation could be added here (e.g., dimensions)

    return { valid: true };
}

/**
 * Compresses an image if needed
 */
export function compressImage(file, maxSizeInMB = 1) {
    return new Promise((resolve, reject) => {
        // This is a placeholder for image compression functionality
        // In a production app, you'd use a library like browser-image-compression

        // For now, we'll just pass through the original file
        resolve(file);
    });
}

/**
 * Gets image dimensions
 */
export function getImageDimensions(imageUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            resolve({
                width: img.width,
                height: img.height,
                aspectRatio: img.width / img.height
            });
        };
        img.onerror = () => {
            reject(new Error('Failed to load image'));
        };
        img.src = imageUrl;
    });
}