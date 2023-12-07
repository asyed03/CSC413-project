import os
import numpy as np
from glob import glob
import shutil
from data import create_dir

HAM10000_PATH = "../HAM10000/HAM10000_scp"
HAM10000_LABEL_PATH = "../HAM10000/HAM10000_scp_labels"

TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1

## fetch the HAM10000 data
def fetch_HAM10000_dataset():
    HAM10000_dataset = glob(HAM10000_PATH + "/*")
    HAM10000_segmentation_masks = glob(HAM10000_LABEL_PATH + "/*")

    return (HAM10000_dataset, HAM10000_segmentation_masks)

## split the data into 4 sets, training, validation, testing, and memory assessment
def split_dataset(data, labels):
    create_dir("trainx/")
    create_dir("trainy/")
    create_dir("validationx/")
    create_dir("validationy/")
    create_dir("testx/")
    create_dir("testy/")
    create_dir("memassesx/")
    create_dir("memassesy/")
    
    len_data = len(data)
    train__val_split = len_data * TRAIN_SIZE + (len_data * VALID_SIZE)
    test_split = train__val_split + (len_data * TEST_SIZE)

    split = [int(train__val_split), int(test_split)]
    data_split = np.split(data, split)
    label_split = np.split(labels, split)

    train_val_data = data_split[0]
    test_data = data_split[1]
    train_val_labels = label_split[0]
    test_labels = label_split[1]

    combined_data_and_labels = list(zip(train_val_data, train_val_labels))
    np.random.shuffle(combined_data_and_labels)

    train_val_data, train_val_labels = zip(*combined_data_and_labels)

    train_split, val_split, mem_split = 0.4, 0.8, 1.0
    train_val_data_len = len(train_val_data)
    train_val_data_split = [int(train_val_data_len * train_split), int(train_val_data_len * val_split), int(train_val_data_len * mem_split)]

    # Take only first three because fourth element is empty list for some reason
    split_data = np.split(train_val_data, train_val_data_split)[:3]
    split_labels = np.split(train_val_labels, train_val_data_split)[:3]

    train_data, val_data, mem_asses_data = split_data
    train_labels, val_labels, mem_asses_labels = split_labels


    print("COPYING TRAINING DATA")
    for train_entry, train_label in zip(train_data, train_labels):
        data_src = train_entry
        label_src = train_label
        entry_filename = os.path.basename(train_entry)
        label_filename = os.path.basename(train_label)

        data_dst = "trainx/" + entry_filename
        label_dst = "trainy/" + label_filename

        shutil.copyfile(data_src, data_dst)
        shutil.copyfile(label_src, label_dst)
    print("FINISHED COPYING TRAINING DATA")

    print("COPYING VALIDATION DATA")
    for val_entry, val_label in zip(val_data, val_labels):
        data_src = val_entry
        label_src = val_label
        entry_filename = os.path.basename(val_entry)
        label_filename = os.path.basename(val_label)
        data_dst = "validationx/" + entry_filename
        label_dst = "validationy/" + label_filename

        shutil.copyfile(data_src, data_dst)
        shutil.copyfile(label_src, label_dst)
    print("FINISHED COPYING VALIDATION DATA")


    print("COPYING TEST DATA")
    for test_entry, test_label in zip(test_data, test_labels):
        data_src = test_entry
        label_src = test_label
        entry_filename = os.path.basename(test_entry)
        label_filename = os.path.basename(test_label)
        data_dst = "testx/" + entry_filename
        label_dst = "testy/" + label_filename

        shutil.copyfile(data_src, data_dst)
        shutil.copyfile(label_src, label_dst)
    print("FINISHED COPYING TEST DATA")

    print("COPYING MEMORY ASSESSMENT DATA")
    for mem_entry, mem_label in zip(mem_asses_data, mem_asses_labels):
        data_src = mem_entry
        label_src = mem_label
        entry_filename = os.path.basename(mem_entry)
        label_filename = os.path.basename(mem_label)
        data_dst = "memassesx/" + entry_filename
        label_dst = "memassesy/" + label_filename

        shutil.copyfile(data_src, data_dst)
        shutil.copyfile(label_src, label_dst)
    print("FINISHED COPYING MEMORY ASSESSMENT DATA")


if __name__ == '__main__':
    (dataset, labels) = fetch_HAM10000_dataset()
    dataset.sort()
    labels.sort()
    split_dataset(dataset, labels)

