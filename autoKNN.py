import os
import sys
import fnmatch
import time

######################################################################################################
### this file should be in 'network/[snapshotsFolder]/'
### each moment of time if checks if there is a ceffemodel is near it
###     if there is one, it runs prediction
###     if there are many, it takes newest. note: it assumes that all caffemodels are have same prefix. newest=largest number in the end
### if after last model appeared 2h past, stop execution
### results are written to result_file: 1NN , 1NN for train data and validation data

# relative to repo path
TRAIN_IMAGES_PATH = 'data/train_images_14/'
TEST_IMAGES_PATH = 'data/test_images/'
MEAN_FILE = 'data/train_mean.bin'

result_file = '_log_autoKNN.txt'
######################################################################################################
repo_path = os.path.abspath(os.getcwd() + '/../../')
sys.path.insert(0, repo_path)
from predictWriter import runPrediction
snap_path = os.getcwd()

last_model = ''
TRAIN_IMAGES_PATH = os.path.join(repo_path, TRAIN_IMAGES_PATH)
TEST_IMAGES_PATH = os.path.join(repo_path, TEST_IMAGES_PATH)
MEAN_FILE = os.path.join(repo_path, MEAN_FILE)


with open(result_file, "w+") as myfile:
    myfile.write("\n-------------------------------------\n")
    myfile.write('TRAIN_IMAGES_PATH ' + TRAIN_IMAGES_PATH)
    myfile.write("\n")
    myfile.write('TEST_IMAGES_PATH' + TEST_IMAGES_PATH)
    myfile.write("\n")


time_when_new_model_appeared = time.time()
while (True):

    ############### get newest caffe model
    list_caffemodels = []
    prototxt_file = ''
    for file in os.listdir('.'):
        if fnmatch.fnmatch(file, '*.caffemodel'):
            list_caffemodels.append(file)
            #print file
        elif fnmatch.fnmatch(file, '*.prototxt'):
            prototxt_file = file
            
    if (len(list_caffemodels) == 0):
        newest_caffemodel_file = ''
    else:
        # all models have same prefix -> last is the newest, because all items in listdir are sorted by name
        newest_caffemodel_file = list_caffemodels[-1]

    ############## check if we did not run that model
    if (last_model == newest_caffemodel_file):
        # no model appeared, wait
        time.sleep(3)
        time_passed_after_last_model_appeared = time.time() - time_when_new_model_appeared
        minutes_passed = time_passed_after_last_model_appeared / (60)
        print 'No new model for ', minutes_passed, 'minutes'
        if (minutes_passed > 2 * 60):
            # stop execution if 2 hours passed
            print '------ Stopped -------'
            break
        else:
            # continue waiting
            continue
            
    ############### did not run the model. 
    print 'Newest caffemodel - ', newest_caffemodel_file
    print prototxt_file
    last_model = newest_caffemodel_file
    time_when_new_model_appeared = time.time()

    PRETRAINDED_MODEL_FILE = os.path.join(os.getcwd(), newest_caffemodel_file)
    MODEL_PROTO_FILE = os.path.join(os.getcwd(), prototxt_file)

    os.chdir('../../')
    (tr1NN, tr3NN, val1NN, val3NN) = runPrediction(PRETRAINDED_MODEL_FILE, MODEL_PROTO_FILE, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, MEAN_FILE)

    # change path back after running
    os.chdir(snap_path)
    
    ##################### write results to file
    with open(result_file, "a") as myfile:
        myfile.write("\n")
        myfile.write(newest_caffemodel_file)
        myfile.write("\n")
        myfile.write('Training data:\n')
        myfile.write('   1NN - ' + str(tr1NN))
        myfile.write("\n")
        myfile.write('   3NN - ' + str(tr3NN))
        myfile.write("\n")
        myfile.write('Validation data:\n')
        myfile.write('   1NN - ' + str(val1NN))
        myfile.write("\n")
        myfile.write('   3NN - ' + str(val3NN))
        myfile.write("\n")
    
with open(result_file, "a") as myfile:
    myfile.write("\n-------------------------------------")