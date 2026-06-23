# AutoML Exam - SS26 (Text Data)

This repo serves as a template for the exam assignment of the AutoML SS26 course
at the university of Freiburg.

The aim of this repo is to provide a minimal installable template to help you get up and running.

For test results on _final dataset_ refer [here](#Running-auto-evaluation-on-test-dataset).

## Installation

To install the repository, first create an environment of your choice and activate it. 

For example, using `venv`:

**Virtual Environment**

```bash
python3 -m venv automl-text-env
source automl-text-env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n automl-text-env python=3.10
conda activate automl-text-env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

*NOTE*: this is an editable install which allows you to edit the package code without requiring re-installations.

You can test that the installation was successful by running the following command:

```bash
python -c "import automl; print(automl.__file__)"
# this should print the full path to your cloned install of this repo
```

We make no restrictions on the python library or version you use, but we recommend using python 3.10 or higher.

## Code

We provide the following:
* `download-datasets.py`: This script downloads all the datasets (both the practice datasets and the official exam dataset) provided for the exam. All datasets are released from the beginning of the exam.

* `run.py`: A script that trains an _AutoML-System_ on the training split of a given dataset and 
  then generates predictions for the test split, saving those predictions to a file. 
  For the training datasets, the test splits will contain the ground truth labels, but for the 
  test dataset the labels of the test split will not be available. 
  You will be expected to generate these labels yourself and submit them to us through GitHub classrooms.

* `automl`: This is a python package that will be installed above and contain your source code for whatever
  system you would like to build. We have provided a dummy `AutoML` class to serve as an example.

*You are completely free to modify, install new libraries, make changes and in general do whatever you want with the code.* 
The *only requirement* for the exam will be that you can generate predictions for the test splits of our datasets in a `.npy` file that we can then use to give you a test score through GitHub classrooms.


## Data

We selected 5 (4 datasets for Phase 1, and 1 dataset for Phase 2) different text-classification datasets which you can use to develop your AutoML system and we provide you with a test dataset to evaluate your system.

You can download all datasets (both the practice datasets and the exam dataset) using:
```bash
python download-datasets.py
```

The dataset can also be manually downloaded and extracted from:

Phase 1: https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-text/text-phase1.zip

Phase 2: https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-text/text-phase2.zip

The downloaded datasets from both the phases should finally have the following structure:
```bash
<data-path>
├── ag_news
│   ├── train.csv
│   ├── test.csv
├── amazon
│   ├── train.csv
│   ├── test.csv
├── imdb
│   ├── train.csv
│   ├── test.csv
├── dbpedia
│   ├── train.csv
│   ├── test.csv
├── yelp
│   ├── train.csv
│   ├── test.csv
```

### Meta-data for datasets:

The following table will provide you an overview of their characteristics and also a reference value for the test accuracy.
*NOTE*: These scores were obtained through a rather simple HPO on a crudely constructed search space, for an undisclosed HPO budget and compute resources.

| Dataset Name | Labels | Rows | Seq. Length: `min` | Seq. Length: `max` | Seq. Length: `mean` | Seq. Length: `median` | Reference Accuracy |
| --- | --- |  --- |  --- |  --- | --- | --- | --- |
| amazon | 3 | 24985 | 4 | 15521 | 512 | 230 | 81.799% |
| imdb | 2 | 25000 | 52 | 13584 | 1300 | 962 | 86.993% |
| ag_news | 4 | 120000 | 99 | 1012 | 235 | 231 | 90.265% |
| dbpedia | 14 | 560000 | 11 | 13573 | 300 | 301 | 97.882% |
| *yelp* | 5 | 650000 | 1 | 5637 | 729 | 537 | 62.082% |

*NOTE*: sequence length calculated at the raw character level

The final test dataset is `yelp`.
It is in the same format as the training datasets, but `test.csv` will only contain `nan`'s for labels.
We expect you to generate these test dataset labels using your pipeline and upload the list of predicted labels to the test branch and get your test score as discussed [here](#Running-auto-evaluation-on-test-dataset).

## Running an initial test

After having downloaded and extracted the data at a suitable location, this is the parent data directory. </br>
To run a quick test:

```bash
python run.py \
  --data-path <data-path> \
  --dataset amazon \
  --epochs 1 \
  --data-fraction 0.2
```
*TIP*: play with the batch size and different approaches for an epoch (or few mini-batches) to estimate compute requirements given your hardware availability.

You are free to modify these files and command line arguments as you see fit.


## Running auto evaluation on test dataset

Important: Auto evaluation on the test dataset will become available only after the start of Phase II on July 21, 2026, at 00:00 CET. Pushes to the test branch made before this time will not be autograded.

Autoevaluation only activates on push to the `test` branch. It is important to note that Github Classroom creates unrelated histories for the `main` branch and `test` branch, that is why you can not use `git merge main` from the `test` branch directly. There are many ways to move the changes from other branches (e.g. from the `main` branch) to the `test` branch even though the commit histories between the branches are unrelated. Here is a simple way:

- Copy your predictions.npy for the corresponding seed (for e.g. from `results/dataset=yelp/seed=42`) to `data/exam_dataset/`.

```bash
# on some_branch (e.g. main) do:
git add data/exam_dataset/predictions.npy
git commit -m "Generated predictions for test data"
git checkout test
#now you should be in the test branch
git checkout main -- data/exam_dataset/predictions.npy # only copies the data/exam_dataset/predictions.npy to the test branch from the mentioned branch and stages it, ready to be comitted
git status # ensure that your latest `.data/exam_dataset/predictions.npy` is staged
git commit -m "Generated predictions for test data, ready for evaluation"
git push
# wait for some time (few seconds) or monitor the web UI of Github to see if the job ran successfully
git pull 
# test scores will be downloaded under `.data/exam_dataset/test_out/` if the job ran successfully
```

Feel free to use any other command to move the prediction files from other branches with unrelated histories to the test branch (`rebase`,`merge some_branch_with_unrelated_history --allow-unrelated-histories`, `stash`...), **<span style="color:red">just make sure that there is nothing else inside `data/exam_dataset/` except for `predictions.npy` and the evaluation results that we push</span>**.

A summary of the evaluation workflow:
* To initialize auto-evaluation for the test data, checkout to the `test` branch.
* Make sure you have named the prediction file `predictions.npy` and placed it in the `data/exam_dataset/` directory in this branch.
* After pushing to it, the evaluation script will be automatically triggered.
* The results are also pushed to your repo (don't forget to `git pull`)
* If no new commits are pulled by `git pull`, check the errors in the Github's `Action` section (red cross inline, last commit message, test branch)

## Final submission

The following must be submitted by `August 3, 2026, 23:59 CET` for a successful project submission and poster participation:

#### **1) Poster submission**
Upload your poster as a PDF file named as `final_poster_text_<team-name>.pdf`, following the template given [here](https://docs.google.com/presentation/d/1T55GFGsoon9a4T_oUm4WXOhW8wMEQL3M/edit?slide=id.p1#slide=id.p1).

#### **2) Test predictions**
The final test predictions should be uploaded in a file `final_test_preds.npy`, with each line containing the predictions for the input in the exact order of `X_test` given.

#### **3) Reproducibility instructions**
TL;DR: Code and instructions to _reproduce_ the above test predictions.

A `run_instructions.md` file that guides through the command to run the designed AutoML solution on the training set of the *final-test-dataset*.
This command should return either a: (i) hyperparameter configuration, (ii) a partially trained model on a hyperparameter configuration, or (iii) a fully trained model in `24 hours` at most.
A second command that given (i), (ii), or (iii) would do the needful that yields predictions for `test_X` for the *final-test-dataset*. This is the `final_test_preds.npy`.

#### **4) Team information**
Upload a file `team_info.txt` with the list of matriculation IDs of team members (*NO NAMES*). (E.g.: 1234567, 7654321)

### Submission checklist:
- [ ] Poster
- [ ] Test predictions
- [ ] Reproducibility instructions
- [ ] Team info
- [x] *Example to denote task being done*

## Tips

* If you need to add dependencies that you and your teammates are all on the same page, you can modify the
  `pyproject.toml` file and add the dependencies there. This will ensure that everyone has the same dependencies

* Please feel free to modify the `.gitignore` file to exclude files generated by your experiments, such as models,
  predictions, etc. Also, be friendly teammate and ignore your virtual environment and any additional folders/files
  created by your IDE.
