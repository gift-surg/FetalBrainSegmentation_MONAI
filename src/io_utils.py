# Copyright 2020 Marta Bianca Maria Ranzini and contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##############
# Basic helpers to create list of img-labels dictionaries for MONAI data loader
# Given a txt file containing the subject IDs for a specific set (e.g. training or validation) and a list of folders to
# search, the functions identify existing files and save them into a dictionary with their full path
import os


# functions to create training and validation lists
def search_file_in_folder_list(folder_list, file_name):
    """
    search a file with a part of name in a list of folders
    :param folder_list: a list of folders
    :param file_name: a substring of a file
    :param output: the full file name
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if os.path.isfile(full_file_name):
            file_exist = True
            break
    if file_exist == False:
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name


def create_data_list(data_folder_list, subject_list, img_postfix='_Image', label_postfix='_Label', is_inference=False):
    """
    create list of all file paths
    :param data_folder_list: list of directories to search
    :param subject_list: list of subject prefix to search (expected filename: <subject_prefix><postfix>.nii.gz)
    :param img_postfix: postfix for image filenames
    :param label_postfix: postfix for label filenames
    :param is_inference: boolean, if set to True, it will search for images only, not for labels
    :return list of paths to existing files with matched name
    """

    if isinstance(data_folder_list, str):
        data_folder_list = [data_folder_list]
    if isinstance(subject_list, str):
        subject_list = [subject_list]

    full_list = []
    for scan_list in subject_list:
        with open(scan_list) as f:
            for line in f:
                subject = line.rstrip()
                image_basename = "{}{}.nii.gz".format(subject, img_postfix)
                image_filename = search_file_in_folder_list(data_folder_list, image_basename)
                label_basename = "{}{}.nii.gz".format(subject, label_postfix)
                label_filename = search_file_in_folder_list(data_folder_list, label_basename)
                if os.path.isfile(image_filename):
                    if is_inference:
                        full_list.append({'img': image_filename})
                    else:
                        if os.path.isfile(label_filename):
                            full_list.append({'img': image_filename, 'seg': label_filename})
                        else:
                            raise IOError('Expected label file: {} not found'.format(label_filename))
                else:
                    raise IOError('Expected image file: {} not found'.format(image_filename))
    return full_list