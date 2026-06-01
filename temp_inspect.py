import importlib.util, pathlib, inspect
path=pathlib.Path("ml_engine.py").resolve()
spec=importlib.util.spec_from_file_location("ml_engine_test", path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('loaded file:', mod.__file__)
print(inspect.getsource(mod.MachineLearningEngine._build_feature_rows))
