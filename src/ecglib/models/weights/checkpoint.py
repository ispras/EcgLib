from typing import Dict, List


class ModelChekpoint(dict):
    @classmethod
    def make_checkpoint(cls, model_info: Dict, exclude_keys: List = None):
        """Make ModelChekpoint instance from input dictionary

        :param model_info: _description_
        :type model_info: Dict
        :param exclude_keys: _description_, defaults to None
        :type exclude_keys: List, optional
        :return: _description_
        :rtype: _type_
        """
        if exclude_keys:
            assert isinstance(exclude_keys, list)
            for key in exclude_keys:
                del model_info[key]

        return cls(model_info)
