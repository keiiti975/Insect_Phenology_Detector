import os
from os.path import join as pj
import re


class Logger(object):
    """
        Save Experiment Args
        Args need
            - experiment_name
            - model_root
    """

    def __init__(self, args, filename="args.txt"):
        self.args = args
        self.file_path = pj(args.model_root, filename)
        if os.path.exists(args.model_root) is False:
            os.makedirs(args.model_root)

    def write(self, msg):
        if self.file_path is not None:
            with open(self.file_path, "a") as f:
                f.write(msg)

    def generate_args_map(self):
        args_keys_list = list(self.args.__dict__.keys())
        args_values_list = list(self.args.__dict__.values())

        pattern = r"__"
        refined_args_map = {}
        for i, args_key in enumerate(args_keys_list):
            is_meta = re.match(pattern, args_key)
            if is_meta is None:
                refined_args_map.update(
                    {args_keys_list[i]: args_values_list[i]})
        return refined_args_map

    def save(self):
        args_map = self.generate_args_map()
        self.write("\nTraining on: " + self.args.experiment_name + "\n")
        self.write("Using the specified args:" + "\n")
        for k, v in args_map.items():
            self.write(str(k) + ": " + str(v) + "\n")