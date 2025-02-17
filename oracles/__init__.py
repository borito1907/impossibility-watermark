# from oracles.prometheus.absolute import PrometheusAbsoluteOracle
# from oracles.prometheus.relative import PrometheusRelativeOracle
from oracles.guidance.rank import RankOracle
from oracles.guidance.joint import JointOracle
from oracles.guidance.solo import SoloOracle
from oracles.guidance.relative import RelativeOracle
from oracles.guidance.binary import BinaryOracle
from oracles.guidance.mutation import MutationOracle
from oracles.guidance.mutation1 import Mutation1Oracle
from oracles.guidance.example import ExampleOracle
from oracles.guidance.diff import DiffOracle
from oracles.rewardbench.armorm import ArmoRMOracle
from oracles.rewardbench.internlm import InternLMOracle
from oracles.rewardbench.offsetbias import OffsetBiasOracle
from oracles.rewardbench.nicolai import NicolAIOracle
from oracles.human.human import HumanOracle