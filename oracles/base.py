from abc import ABC, abstractmethod
from guidance import models
from dotenv import load_dotenv

# Abstract base class for all oracles
class Oracle(ABC):
    
    judge = None

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg # config.oracle_args

    def _initialize_llm(self):
        log.info("Initializing a new Oracle model from cfg...")
        if "gpt-" in self.cfg.model_id:
            load_dotenv()
            llm = models.OpenAI(
                self.cfg.model_id,
                echo=False
            )
        else:
            llm = models.Transformers(
                self.cfg.model_id, 
                echo=False,
                cache_dir=self.cfg.model_cache_dir, 
                device_map=self.cfg.device_map
            )
        return llm
    
    @abstractmethod
    def evaluate(self, instruction, output_1, output_2, **kwargs):
        pass

    @abstractmethod
    def extract_label(self, evaluation):
        pass

    @abstractmethod
    def test(self, instruction, output_1, output_2, label, **kwargs):
        pass
    