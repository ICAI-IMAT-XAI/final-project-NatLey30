## Running the Project Locally (Docker Compose)
### Steps

1. Clone the repository:
```bash
git clone <repository_url>
cd final-project-NatLey30
```

2. Train the model
```bash
python src/train.py
```

3. Build images and containers
```bash
docker build -t natley30/api -f Dockerfile.api .
docker build -t natley30/web -f Dockerfile.web .
docker-compose up
```

### Notes on Docker Images
The Docker image names used (natley30/api and natley30/web) correspond to my Docker Hub account.

If you wish to use your Docker Hub account (to push the images) or change the name, you should:

1. build the images using your namespace and image names.
```bash
docker build -t <your_username>/<api_name> -f Dockerfile.api .
docker build -t <your_username>/<web_name> -f Dockerfile.web .
```

2. update the image names accordingly in docker-compose.yaml.