from typing import List, Tuple

import reasoning_gym as rg
from datasets import Dataset
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset

from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric


class ReasoningGymEnv(SingleTurnEnv):
    def __init__(
        self,
        gym: str | List[str | dict],
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        seed: int = 0,
        **kwargs,
    ):
        self.gym = gym
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        total_examples = num_train_examples + num_eval_examples
        self.rg_dataset = self.build_rg_dataset(gym, total_examples, seed=seed)
        dataset, eval_dataset = self.rg_to_hf(self.rg_dataset)
        parser = XMLParser(fields=["think", "answer"])
        rubric = Rubric(parser=parser)

        def check_answer_reward_func(completion, answer, **kwargs) -> float:
            entry = self.rg_dataset[answer]
            response = str(parser.parse_answer(completion)).strip()
            reward = self.rg_dataset.score_answer(answer=response, entry=entry)
            return reward

        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)
        system_prompt = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"]  # type: ignore
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )
        self.parser = parser
        self.rubric = rubric

    def build_rg_dataset(
        self, gym: str | List[str | dict], total_examples: int = 1000, seed: int = 0
    ) -> ProceduralDataset:
        if isinstance(gym, str):
            return rg.create_dataset(gym, size=total_examples, seed=seed)
        dataset_specs = []
        for dataset_config in gym:
            if isinstance(dataset_config, str):
                dataset_specs.append(
                    DatasetSpec(name=dataset_config, weight=1.0, config={})
                )
            elif isinstance(dataset_config, dict):
                dataset_specs.append(DatasetSpec(**dataset_config))
            else:
                raise ValueError(f"Invalid dataset config: {dataset_config}")
        return rg.create_dataset(
            "composite", datasets=dataset_specs, size=total_examples, seed=seed
        )

    def rg_to_hf(self, rg_dataset: ProceduralDataset) -> Tuple[Dataset, Dataset]:
        train_dataset_rows = []
        eval_dataset_rows = []
        for i, x in enumerate(rg_dataset):
            row = {
                "question": x["question"],
                "answer": i,
                "task": x["metadata"]["source_dataset"],
            }
            if i < self.num_train_examples:
                train_dataset_rows.append(row)
            else:
                eval_dataset_rows.append(row)
        dataset = Dataset.from_list(train_dataset_rows)
        eval_dataset = Dataset.from_list(eval_dataset_rows)
        return dataset, eval_dataset


def load_environment(
    gym: str | List[str | dict] = "arc_1d",
    num_train_examples: int = 2000,
    num_eval_examples: int = 2000,
    **kwargs,
):
    vf_env = ReasoningGymEnv(
        gym=gym,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        **kwargs,
    )
    return vf_env
