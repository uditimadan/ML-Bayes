#@author: uditi madan
import random
import sys


## ZeroR - our first learning algorithm.

def zeroR(data):


  # Skip the header line (assuming data[0] is header)
  classifications = [line.strip().split(',')[-1] for line in data[1:]]

  # Returns the most frequent classification (mode)
  return max(set(classifications), key=classifications.count)


def randR(data):

  # Skip the header line
  classifications = [line.strip().split(',')[-1] for line in data[1:]]

  # Returns a random classification from the list
  return random.choice(classifications)


## Load data from within the script
data_file = "breast-cancer.data"
try:
  with open(data_file, 'r') as f:
    data = f.readlines()
except FileNotFoundError:
  print(f"Error: File '{data_file}' not found!")
  exit(-1)

## Our main

classify_type = "-z"  # Default classification type (ZeroR)
if len(sys.argv) > 1:
  classify_type = sys.argv[1]
  if classify_type not in ("-z", "-r"):
    print("Usage: python ZeroR.py {-z|-r}")
    exit(-1)

if classify_type == "-z":
  print(zeroR(data))
else:
  print(randR(data))
