# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IWSLT 2017 dataset """


import os

import datasets


_HOMEPAGE = "https://sites.google.com/site/iwsltevaluation2017/TED-tasks"

_DESCRIPTION = """\
The IWSLT 2017 Multilingual Task addresses text translation, including zero-shot translation, with a single MT system across all directions including English, German, Dutch, Italian and Romanian. As unofficial task, conventional bilingual text translation is offered between English and Arabic, French, Japanese, Chinese, German and Korean.
"""

_CITATION = """\
@inproceedings{cettolo-etal-2017-overview,
    title = "Overview of the {IWSLT} 2017 Evaluation Campaign",
    author = {Cettolo, Mauro  and
      Federico, Marcello  and
      Bentivogli, Luisa  and
      Niehues, Jan  and
      St{\\"u}ker, Sebastian  and
      Sudoh, Katsuhito  and
      Yoshino, Koichiro  and
      Federmann, Christian},
    booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",
    month = dec # " 14-15",
    year = "2017",
    address = "Tokyo, Japan",
    publisher = "International Workshop on Spoken Language Translation",
    url = "https://aclanthology.org/2017.iwslt-1.1",
    pages = "2--14",
}
"""

REPO_URL = "https://hf-mirror.com/datasets/bbaaaa/iwslt14-de-en-preprocess/resolve/main/"
URL = REPO_URL + "data/de-en.zip"


class IWSLT2017Config(datasets.BuilderConfig):
    """BuilderConfig for NewDataset"""

    def __init__(self, pair, **kwargs):
        """

        Args:
            pair: the language pair to consider
            is_multilingual: Is this pair in the multilingual dataset (download source is different)
            **kwargs: keyword arguments forwarded to super.
        """
        self.pair = pair
        super().__init__(**kwargs)


class IWSLT2017(datasets.GeneratorBasedBuilder):
    """The IWSLT 2017 Evaluation Campaign includes a multilingual TED Talks MT task."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = IWSLT2017Config
    BUILDER_CONFIGS = [
        IWSLT2017Config(
            name="de-en",
            description="A small dataset",
            version=datasets.Version("1.0.0"),
            pair='de-en',
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"translation": datasets.features.Translation(languages=self.config.pair.split("-"))}
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        source, target = self.config.pair.split("-")
        bi_url = URL
        dl_dir = dl_manager.download_and_extract(bi_url)
        data_dir = os.path.join(dl_dir, f"{source}-{target}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"train.{source}",
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"train.{target}",
                        )
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"test.{source}",
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"test.{target}",
                        )
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"valid.{source}",
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"valid.{target}",
                        )
                    ],
                },
            ),
        ]

    def _generate_examples(self, source_files, target_files):
        """Yields examples."""
        id_ = 0
        source, target = self.config.pair.split("-")
        for source_file, target_file in zip(source_files, target_files):
            with open(source_file, "r", encoding="utf-8") as sf:
                with open(target_file, "r", encoding="utf-8") as tf:
                    for source_row, target_row in zip(sf, tf):
                        source_row = source_row.strip()
                        target_row = target_row.strip()

                        if source_row.startswith("<"):
                            if source_row.startswith("<seg"):
                                # Remove <seg id="1">.....</seg>
                                # Very simple code instead of regex or xml parsing
                                part1 = source_row.split(">")[1]
                                source_row = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                target_row = part1.split("<")[0]

                                source_row = source_row.strip()
                                target_row = target_row.strip()
                            else:
                                continue

                        yield id_, {"translation": {source: source_row, target: target_row}}
                        id_ += 1
