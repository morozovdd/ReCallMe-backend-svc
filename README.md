```markdown
# ReCallMe API

ReCallMe is a Flask-based REST API wrapper for interacting with various Large Language Models (LLMs) such as Gemini and Anthropic. This API provides endpoints for text and image-based inferences, as well as conversation handling with memory, allowing developers to integrate LLM capabilities into their applications.


## Table of Contents

* [Installation](#installation)
* [Environment Variables](#environment-variables)
* [API Documentation](#api-documentation)
* [Endpoints](#endpoints)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)


## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/recallme.git
cd recallme
```

2. Install dependencies using Poetry:

Make sure Poetry is installed.

```bash
poetry install
```

3. Set up the environment:

Ensure necessary environment variables are set up (see below).

4. Run the application:

```bash
poetry run flask run --host=0.0.0.0 --port=8080
```

5. Access the Swagger UI documentation:

Navigate to `http://localhost:8080/recall-svc/api/v1` to view interactive API documentation.


## Environment Variables

Create a `.env` file in the project root with the following variables:

```
ANTHROPIC_KEY=your_anthropic_api_key
# Add other API keys as needed (e.g., for Gemini)
```

Replace `your_anthropic_api_key` with the actual API key for the Anthropic LLM service.


## API Documentation

ReCallMe uses Flask-RESTx for the API framework and Swagger UI for interactive documentation. The documentation can be accessed at `http://localhost:8080/recall-svc/api/v1/swagger`.


## Endpoints


### 1. `/multiply`

* **Description:** Multiplies two numbers provided in the request.
* **Method:** `POST`
* **Payload:**
```json
{
  "a": 2,
  "b": 3
}
```
* **Response:**
```json
6
```

### 2. `/gemini-text-inference`

* **Description:** Makes a text inference call to the Gemini LLM.
* **Method:** `POST`
* **Payload:**
```json
{
  "query": "What is the capital of France?"
}
```
* **Response:**
```json
{
  "response": "The capital of France is Paris."
}
```

### 3. `/gemini-chat-with-history`

* **Description:** A conversational endpoint that maintains chat history and supports custom prompts.
* **Method:** `POST`
* **Payload:**
```json
{
  "query": "Tell me a joke.",
  "custom_prompt": "Your message:",
  "history": []
}
```
* **Response:**
```json
{
  "response": "Why did the chicken cross the road? To get to the other side!",
  "history": [...]
}
```


### 4. `/anthropic-text-inference`

* **Description:** Makes a text inference call to the Anthropic LLM.
* **Method:** `POST`
* **Payload:**
```json
{
  "query": "Summarize the history of AI."
}
```
* **Response:**
```json
{
  "response": "AI has evolved significantly from early rule-based systems to today's deep learning models..."
}
```


### 5. `/gemini-media-inference`

* **Description:** Uses Gemini LLM to analyze images with optional text input and history.
* **Method:** `POST`
* **Payload:**  (How the `file` parameter is structured depends on your implementation - likely multipart/form-data)
```json
{
  "file": "<image_file>",
  "input_text": "Describe this image.",
  "mime_type": "image/png",
  "history": []
}
```
* **Response:**
```json
{
  "response": "This is an image of a cat.",
  "history": [...]
}
```


## Usage

To start the server and use the endpoints, follow these steps:

1. Ensure the server is running on `0.0.0.0` so it can be accessed externally.
2. Use tools like `curl`, Postman, or Swagger UI to interact with the API endpoints.


### Example Request with curl

```bash
curl -X POST "http://localhost:8080/recall-svc/api/v1/multiply" -H "Content-Type: application/json" -d '{"a":5,"b":6}'
```


## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Make your changes and submit a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
```


This improved README provides clearer formatting, more detailed explanations, and addresses potential issues like the file upload in the `/gemini-media-inference` endpoint.  Remember to replace `yourusername` with your actual GitHub username.  Also, create a LICENSE file (typically MIT License) in the project root.
