# Sentiment Analysis Flask App
This repository contains a sentimnent analysis web app using flask. Model is based on a finetuned version of BERT language model previusly trained on tweeter sentiment dataset. Details of training and model can be found in this repository. 
Web app allows user to enter text and predicts the sentiment (positive , negative or neutral) associated with it.
Docker container is added used to provide containerized environment that is consistent across different machines and platforms and easy deployment.
## Model State and Checkpoint:
Model state files and model checkpoint are not save in this repository due to the large file sizes, All requiured models state and model checkpoints will be downloaded directly in the colab notebooks. Additionaly the best model state can be access [here](https://drive.google.com/file/d/1afvHvYRK2qvOMk-oVF6KDYrIO6hpUAik/view?usp=sharing) and the latest model check point can be downloaded from [here](https://drive.google.com/file/d/1alaDfFsBbJ9WiTkKFdKkERCfb022Ai7E/view?usp=sharing).

## Prerequisites

- Docker: Ensure that you have Docker installed on your machine. You can download and install Docker from the [official website](https://www.docker.com/get-started).

## Getting Started

Follow these steps to run the Sentiment Analysis Demo:

1. Clone this repository to your local machine:
   ```shell
   git clone https://github.com/sabhaee/sentiment-analysis-flask-app.git

2. Download and save the pre-trained model into the `api/model` folder.

3. Change into the project directory:
    ```shell
    cd sentiment-analysis-flask-app

3. Build the Docker image:
    ```shell
    docker build -t sentiment-analysis-demo  ./api

4. Run the Docker container:
    ```shell
    docker run -p 5000:5000 sentiment-analysis-demo

5. Open a web browser and go to `http://localhost:5000` to access the application.

## Usage

1. Enter some text in the input field.

2. Click the "Analyze" button to perform sentiment analysis.

3. The application will display the entered text, the sentiment analysis result, and a bar chart visualizing the sentiment scores.

## Sample output

![Screenshot](https://github.com/sabhaee/sentiment-analysis-flask-app/blob/main/images/Screenshot.png)


## Customization

- To modify the styling of the application, you can edit the `static/styles.css` file.

- To customize the sentiment analysis model or algorithm, you can modify the code in the Flask route (`app.py`) and (`model.py`) in the api directory responsible for sentiment analysis.

- Additional functionality and enhancements can be implemented by extending the existing codebase.
