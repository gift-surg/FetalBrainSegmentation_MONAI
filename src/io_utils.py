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


def create_data_list(data_folder_list, subject_list, img_postfix='_Image', label_postfix='_Label',
                     mask_postfix=None, is_inference=False):
    """
    create list of all file paths
    :param data_folder_list: list of directories to search
    :param subject_list: list of subject prefix to search (expected filename: <subject_prefix><postfix>.nii.gz)
    :param img_postfix: postfix for image filenames
    :param label_postfix: postfix for label filenames
    :param mask_postfix: postfix for mask filenames
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
                if mask_postfix is not None:
                    mask_basename = "{}{}.nii.gz".format(subject, mask_postfix)
                    mask_filename = search_file_in_folder_list(data_folder_list, mask_basename)
                    assert os.path.isfile(mask_filename), "Expected mask file: {} not found".format(mask_filename)
                else:
                    mask_filename = []
                if os.path.isfile(image_filename):
                    if is_inference:
                        full_list.append({'img': image_filename, 'mask': mask_filename})
                    else:
                        if os.path.isfile(label_filename):
                            full_list.append({'img': image_filename, 'seg': label_filename, 'mask': mask_filename})
                        else:
                            raise IOError('Expected label file: {} not found'.format(label_filename))
                else:
                    raise IOError('Expected image file: {} not found'.format(image_filename))
    return full_list