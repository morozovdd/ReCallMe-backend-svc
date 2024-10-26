#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
REPOSITORY=recallme-repo
SERVICE_NAME=recallme-flask-service
IMAGE_NAME=recallme-flask-app
PORT=8095

echo "Authenticating with Google Cloud..."
gcloud auth configure-docker $REGION-docker.pkg.dev -q

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "Creating Artifact Registry repository if it doesn't exist..."
gcloud artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker repository for RecallMe app" || true

echo "Tagging Docker image..."
docker tag $IMAGE_NAME $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME

echo "Pushing Docker image to Artifact Registry..."
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port $PORT

echo "Deployment successful!"
