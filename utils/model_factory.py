from pydoc import locate
import inspect


def instantiate(_cfg, *args, **kwargs):
    if _cfg is None:
        return None
    
    assert "_target_" in _cfg, "Configuration must contain _target_ field"
    assert isinstance(_cfg._target_, str), "_target_ field must be a string"
    
    cls = locate(_cfg._target_)
    assert cls is not None, f"{_cfg._target_} is not a valid class reference"
    
    expected_args = inspect.getfullargspec(cls)[0][1:]
    expected_kwargs = dict()
    for k, v in _cfg.items():
        if k in expected_args:
            expected_kwargs[k] = v
    
    for k, v in kwargs.items():
        if k in expected_args:
            expected_kwargs[k] = v
            
    return cls(*args, **expected_kwargs)