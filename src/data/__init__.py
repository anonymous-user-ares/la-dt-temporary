"""
Data module for LA-DT experiments.
Contains synthetic and real data generators for GAT model training and evaluation.
"""

from .gat_data_generator import (
    SyntheticDataGenerator,
    SensorGraphDataset,
    create_sensor_graph_fully_connected,
    create_data_loaders,
)

__all__ = [
    "SyntheticDataGenerator",
    "SensorGraphDataset",
    "create_sensor_graph_fully_connected",
    "create_data_loaders",
]
