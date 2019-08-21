# Social Media Analysis

This repository houses the code for social media analysis on Guyanese Facebook pages. 

## Prerequisites  

Python 3.7.3 or greater. If you do not have Python installed, it is recommended to install Anaconda to get Python. Anaconda is an open-source package that includes Python and many popular Python packages. The current version of Anaconda is [here](https://www.anaconda.com/distribution/) 

(Optional) A Git client. If you are on Windows, it is recommended to install Git Bash. Git Bash is a terminal application that includes Git and runs Linux commands on a Windows system. Git Bash can be found [here](https://gitforwindows.org)

## Setup

After installing Python, clone or download this repository. If cloning with Git, the command is:

```bash
git clone https://github.com/czig/nh19_sentiment.git
```

After cloning the repository, install the required Python packages, which are listed in requirements.txt file. An easy way to do this with the Anaconda package is to run the following command in a command prompt/terminal:

```bash
conda install --file requirements.txt
```

If not using Anaconda, use pip to install the packages:

```bash
pip install -r requirements.txt
```

Create the following directory structure at the root of the repository:

```bash
├── bigramCharts
├── lda_vis
├── models
├── numbers
├── tmp
├── topics
├── trigramCharts
├── wordclouds
└── wordFreqPlots
```

In order to pull data from Facebook, the `facebookPull.py` script needs your Facebook access token. To get your access token, log into your Facebook page and navigate to a public Facebook page. Right click on the page and select "View page source" from the drop down menu. A new tab should open, and once in the new tab, press Ctrl+F and search for "access_token". Move through the matches until you find a string that is 187 characters long and is a mix of letters and numbers. This string is your access token. Using a text editor, create a file named "access_token.txt" at the root of the directory. Place your access token, and *only* your access token, in this file and save. Now you will be able to pull Facebook posts and comments with the `facebookPull.py` script.

## Usage

In order to view the usage for each script, run:

```bash
python <<filename>> -h
```

Accepted arguments and default values, along with a description of each argument, will be printed to the terminal. Arguments are handled with the argparse package, and each argument name is preceded by two hyphens and followed by a space. The value for the argument follows the space after the argument name. An example of running one of the scripts is below:

```bash
python generateAllWordCloud.py --type comments --pages guy --start_date 2019-01-01 --end_date 2019-07-29
```

A few of the scripts do not accept arguments, and thus, do not use the argparse package. These scripts can be run with `python <<filename>>`






