"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        Essay_prompt: str,
        Essay: str,
        Response: str = None,
        Grade: Union[None, str] = None
    ) -> str:
        # returns the full prompt from Essay_prompt and optional Essay
        # if a label (=Response, =output) is provided, it's also appended.
        if Response:
            res = self.template["prompt_input"].format(
                Essay_prompt=Essay_prompt, Essay=Essay, Response = Response, Grade = Grade
                )
        else:
            res = self.template["prompt_no_response"].format(
                Essay_prompt=Essay_prompt, Essay=Essay
                )
        # if Grade:
        #     res = f"{res}{Grade}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
