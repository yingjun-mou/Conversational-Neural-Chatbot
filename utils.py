# General util functions
import logging
import os
import pickle
import re


def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)

def print_list(l, K=None):
	# If K is given then only print first K
	for i, e in enumerate(l):
		if i == K:
			break
		print(e)
	print()

def remove_multiple_spaces(string):
	return re.sub(r'\s+', ' ', string).strip()

def save_in_pickle(save_object, save_file):
	with open(save_file, "wb") as pickle_out:
		pickle.dump(save_object, pickle_out)

def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)

def save_in_txt(list_of_strings, save_file):
	with open(save_file, "w") as writer:
		for line in list_of_strings:
			line = line.strip()
			writer.write(f"{line}\n")

def load_from_txt(txt_file):
	with open(txt_file, "r") as reader:
		all_lines = list()
		for line in reader:
			line = line.strip()
			all_lines.append(line)
		return all_lines