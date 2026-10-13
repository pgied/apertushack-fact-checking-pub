"""Which strategies there are: the Catalog finds them in src/strategies/, loads them by name, and makes
them known to Hydra.

A strategy is a folder src/strategies/<name>/ with an empty __init__.py. One of its modules, for
example strategy.py, defines one subclass of Strategy (src/strategy.py). A single file
src/strategies/<name>.py works too. A strategy is found by its name, so nothing else has to change:

    uv run python -m src strategy=<name> task=advanced limit=30

Files and folders whose names start with _ are not strategies: they hold code that strategies share.
The baseline, src/strategies/baseline/, is the template to copy.
"""

import importlib
import pkgutil
import sys
from types import ModuleType

from hydra.core.config_store import ConfigStore

import src.strategies
from src.strategy import Strategy


class Catalog:
    """The strategies in a package: src/strategies/, unless another one is given."""

    def __init__(self, package: ModuleType = src.strategies):
        self.package = package

    def names(self) -> list[str]:
        """The names of the strategies: the folders and files whose names do not start with _."""
        modules = pkgutil.iter_modules(self.package.__path__)
        return sorted(module.name for module in modules if not module.name.startswith("_"))

    def load(self, name: str) -> type[Strategy]:
        """The subclass of Strategy that the strategy `name` defines, in any module of its folder."""
        strategy = importlib.import_module(f"{self.package.__name__}.{name}")
        classes = []
        for module in [strategy, *self.modules_of(strategy)]:
            for value in vars(module).values():
                if self.defines(strategy, value) and value not in classes:
                    classes.append(value)
        if len(classes) != 1:
            raise TypeError(f"src/strategies/{name} has to define one subclass of Strategy, not {len(classes)}")
        return classes[0]

    def register(self) -> None:
        """Make the strategies known to Hydra, as the config group `strategy` with one option each.

        A strategy that does not load is left out, with a message, so that it does not stop the others.
        """
        store = ConfigStore.instance()
        for name in self.names():
            try:
                strategy = self.load(name)
            except Exception as error:
                print(f"Strategy {name} is left out, it does not load: {error!r}", file=sys.stderr)
                continue
            store.store(group="strategy", name=name, node=strategy.Settings(name=name))

    @staticmethod
    def modules_of(strategy: ModuleType) -> list[ModuleType]:
        """The modules in a strategy's folder; none for a strategy that is a single file."""
        if not hasattr(strategy, "__path__"):
            return []
        modules = pkgutil.iter_modules(strategy.__path__)
        return [importlib.import_module(f"{strategy.__name__}.{module.name}") for module in modules]

    @staticmethod
    def defines(strategy: ModuleType, value) -> bool:
        """Whether `value` is a subclass of Strategy defined in the strategy's own modules: a class it only
        imports, such as Strategy itself or the class it builds on, does not count."""
        if not (isinstance(value, type) and issubclass(value, Strategy)):
            return False
        return value.__module__ == strategy.__name__ or value.__module__.startswith(strategy.__name__ + ".")
