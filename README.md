# X-Ray Diagnosis ML Project

A web application for diagnosing X-ray images using machine learning.

## Docker Setup

This project has been dockerized for easy deployment and portability.

### Prerequisites

- Docker
- Docker Compose

### Running with Docker

1. Make sure Docker and Docker Compose are installed on your system.

2. Clone this repository:
   ```
   git clone <repository-url>
   cd xray_diagnosis_ML
   ```

3. Start the application:
   ```
   docker-compose up
   ```

   This will build and start both the frontend and backend services.

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

5. To stop the application:
   ```
   docker-compose down
   ```

### Troubleshooting Docker Issues

If `docker-compose up` gets stuck or shows no progress:

1. Try running with verbose output:
   ```
   docker-compose up --verbose
   ```

2. Build the images separately:
   ```
   docker-compose build --no-cache
   ```

3. Test just the backend service:
   ```
   docker-compose -f docker-compose.debug.yml up
   ```

4. Check Docker logs for errors:
   ```
   docker-compose logs
   ```

5. Make sure the model file path is correct:
   ```
   ls -la src/best_model.pth
   ```

6. If you're on Windows, make sure file paths are using the correct format.

### Important Notes

- The ML model file (`best_model.pth`) must be present in the `src/` directory.
- The API endpoint is configured to use the service name `backend:5000` for container networking.
- No dataset is included in the Docker images.

## Project Structure

- `frontend/`: React.js web application
- `backend/`: Flask API server
- `src/`: Contains ML model and related code

## Development

For development without Docker, refer to the setup instructions in each component's directory. 