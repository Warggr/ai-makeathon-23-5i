# PersonalizedMedicine

## Background

This is a Docker app that can be used with [FeatureCloud.ai](https://featurecloud.ai/).

## Tech stack

The app is based on the [FeatureCloud app blank template](https://github.com/careforrare/PersonalizedMedicine).
As such, it uses the following software inside the Dockerfile:

- nginx server
- Python file `main.py` as an entrypoint, which imports the `bottle` and `api` packages

### building the app docker image
Once app implementation is done, building the docker image for testing or adding it to
[FeatureCloud AI store](https://featurecloud.ai/ai-store?view=store&q=&r=0),
developers should provide the following files.

## Running the application

### Locally
#### Prerequisites
```shell
# Optional: create a virtual environment
pip install virtualenv
python -m venv careforrare
# either
source ./careforrare/bin/activate # For Mac Users
# or
./careforrare/Scripts/Activate.ps1 # For Windows Users (use Powershell)

# Install Requirements
pip install -r requirements.txt
```
#### Running the app
```shell
python src/main.py
```

### On Docker

```shell
make run

# Trigger the start of the application states
curl --location 'http://localhost:9000/setup' --header 'Content-Type: application/json' --data '@test.json'

# Look at logs using. Make sure to close container after testing
docker logs <containerID>
```

# Push the new image to the registry
```shell
pip install featurecloud
make push
```

Alternatively you are free to utilize the full functionalities of the feature-cloud api and Testbed
https://featurecloud.ai/developers

Then either download YOUR_APPLICATION image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/YOUR_APPLICATION
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/YOUR_APPLICATION
```

Please provide example data so others can run YOUR_APPLICATION with the desired settings in the `config.yml` file.

#### Run YOUR_APPLICATION in the test-bed

You can run YOUR_APPLICATION as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also run the app using CLI:

```shell
featurecloud test start --app-image featurecloud.ai/YOUR_APPLICATION --client-dirs './sample/c1,./sample/c2' --generic-dir './sample/generic'
```
