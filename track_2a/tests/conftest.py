"""Shared by the tests: Hydra has to know the strategies before it can compose the configuration."""

from src.catalog import Catalog

Catalog().register()
