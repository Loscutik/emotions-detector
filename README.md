# emotion-detector
## Task's description

The goal of the project is to implement a system that detects the emotion on a face from a webcam video stream.
   - detect a face in an image
   - train a CNN to detect the emotion on a face
   - classify emotions from a video stream

## Links
[subject] (https://github.com/01-edu/public/tree/master/subjects/ai/emotions-detector)
[audit] (https://github.com/01-edu/public/tree/master/subjects/ai/emotions-detector/audit)

## Running 
If you use Python virtual environment, you can set up the one from a file in the `environments` directory.
If you don't work with virtual environments or do not want to, you can use Docker image.
Or you can just read html version of the notebooks which you can find in results/html_notebook directory. 

### Virtual environment 

The virtual environment uses Python >=3.12 with the libraries:
  - jupyter
  - numpy
  - pandas
  - matplotlib
  - tabulate
  - scikit-learn==1.5.1
  - tensorflow==2.17.0
  - keras-3.6.0
  - opencv-python==4.10.0.84

1. Create an identical environment

   - using conda 
       - (creates an environment named 'sklearn'):
       `conda env create -f environment.yml`
       - activate the environment:
       `conda activate sklearn`
   OR
   - using pip:
     `pip install -r requirements.txt`

2. Run the notebook.

  `jupyter lab notebook/nlp_eneriched_news.ipynb`

### Docker
Run a script:
- on Windows:    `docker_run.bat`
- on Linux:      `docker_run.sh`
    
- If you use WSL on Windows 11, you can encounter the error "UtilAcceptVsock:250: accept4 failed 110" (I did :) ).
In this case run 
  - `pre_docker_run_4WSL_win11.sh`
   and then
  - `docker_run.sh`
    
If it is needed, give give execute permissions to the scripts:
- `chmod +x pre_docker_run_4WSL_win11.sh`
- `chmod +x docker_run.sh`

After the container runs, the script will show url to open jupyter in your browser. 
If it will not, in another terminal run `docker exec c-emotions jupyter server list`.

In jupyter on the left side you will see files list.
You can find the notebook `trainig CNN.ipynb` in `notebook` directory.

## Author

Olena Budarahina ([obudarah](https://01.kood.tech/git/obudarah))