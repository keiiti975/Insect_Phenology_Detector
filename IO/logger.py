import os
from os.path import join as pj
import re


class Logger(object):
    """
        引数ロガー
    """

    def __init__(self, args, filename="args.txt"):
        """
            初期化関数
            引数:
                - args: 引数クラス
                model_rootが必須
                - filename: str, 引数を保存するファイルのファイル名
        """
        self.args = args
        self.file_path = pj(args.model_root, filename)
        if os.path.exists(args.model_root) is False:
            os.makedirs(args.model_root)

    def write(self, msg):
        """
            書き込み関数
            引数:
                - msg: str, ファイルに追加する文字列
        """
        if self.file_path is not None:
            with open(self.file_path, "a") as f:
                f.write(msg)

    def generate_args_map(self):
        """
            引数辞書の作成
        """
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
        """
            ファイルの保存
        """
        args_map = self.generate_args_map()
        self.write("\nTraining on: " + self.args.experiment_name + "\n")
        self.write("Using the specified args:" + "\n")
        for k, v in args_map.items():
            self.write(str(k) + ": " + str(v) + "\n")