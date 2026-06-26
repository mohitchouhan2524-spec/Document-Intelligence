from typing import NamedTuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from classifier.features import HandcraftedFeatures


class ClassifierArtifacts(NamedTuple):
    pipeline: Pipeline
    handcrafted: HandcraftedFeatures
    label_encoder: LabelEncoder
    method_map: dict[int, str | None]