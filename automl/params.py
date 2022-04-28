import inspect
import json
import warnings
from copy import deepcopy


class Params:
    def __init__(self, cls, param_file=None, param_str=None, **kwargs):
        input_params = self.parse_file(param_file)
        str_input_params = self.parse_param_str(param_str)
        input_params.update(str_input_params)
        input_params.update(kwargs)

        self.input_params = self.check_input_params(cls, input_params)

    def check_input_params(self, cls, input_params):
        # check and adaptive parameter type for input_params
        default_params = self.load_cls_default_params(cls)
        for key, value in input_params.items():
            parse_func = getattr(self, key, None)
            if parse_func:
                value = parse_func(value)
            elif key in default_params and default_params.get(key) is not None:
                value = self.match_type(default_params[key], value)
            else:
                value = self.eval(value)
            input_params[key] = value
        return input_params

    @staticmethod
    def load_cls_default_params(cls):
        default_values = deepcopy(cls.__init__.__defaults__)
        var_names = cls.__init__.__code__.co_varnames
        var_names = [
            name for name in var_names if name not in {"self", "kwargs"}]
        assert len(default_values) == len(var_names)
        return dict(zip(var_names, default_values))

    @staticmethod
    def match_type(refer_var, input_var):

        if isinstance(refer_var, bool):
            tag = str(input_var).lower()
            if tag not in {"true", "false"}:
                warnings.warn(f"value : {input_var} error, set to False")

            input_var = tag == "true"

        elif refer_var is not None:
            refer_type = type(refer_var)
            input_var = refer_type(input_var)
        return input_var

    @staticmethod
    def eval(value):
        try:
            value = eval(value)
        except:
            value = value
        return value

    @staticmethod
    def parse_file(path):
        path = path or ""
        if not path.strip():
            return {}
        with open(path, "r") as r_f:
            params = json.load(r_f)
        return params

    @staticmethod
    def parse_param_str(param_str):
        param_str = param_str or ""
        if not param_str.strip():
            return {}
        name_value_pairs = param_str.split(";")
        pairs = []
        for name_value_pair in name_value_pairs:
            if not name_value_pair.strip():
                continue
            k_v = name_value_pair.split("=")
            if len(k_v) != 2:
                warnings.warn(f"{name_value_pair} error, will be ignore")
                continue
            key, value = name_value_pair.split("=")
            pairs.append((key.strip(), value.strip()))
        params = dict(pairs)
        return params

    @staticmethod
    def class_weight(value):
        try:
            value = eval(value)
        except:
            value = value
        return value

    def __str__(self):
        input_params_message = str(self.input_params)
        message = f"input_params: {input_params_message}"
        return message

    __repr__ = __str__