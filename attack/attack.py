import logging
import time
from tqdm import tqdm
from utils import (
    save_to_csv,
    length_diff_exceeds_percentage,
    count_num_of_words
)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

class Attack:
    # The watermarker should be in detection mode so we don't waste resources.
    def __init__(self, cfg, mutator, quality_oracle=None, watermarker=None, ):
        self.cfg = cfg
        self.watermarker = watermarker
        self.mutator = mutator
        self.quality_oracle = quality_oracle
        self.results = []
        self.mutated_texts = []
        self.backtrack_patience = 0
        self.patience = 0
        self.max_mutation_achieved = 0
        self.base_step_metadata = {
            "step_num": -1,
            "mutation_num": 0,
            "current_text": "",
            "mutated_text": "", 
            "current_text_len": -1,
            "mutated_text_len": -1, 
            "length_issue": False,
            "quality_analysis" : {},
            "quality_preserved": False,
            "watermark_detected": False,
            "watermark_score": -1,
            "backtrack" : False,
            "timestamp": "",
        }

        self.use_max_steps = self.cfg.attack.use_max_steps
        self.num_steps = self.cfg.attack.max_steps if self.use_max_steps else self.cfg.attack.target_mutations

        if self.cfg.attack.check_quality:
            assert quality_oracle is not None, "cfg.attack.check_quality=True so you must initialize your attack with a quality oracle!"

        if self.cfg.attack.check_watermark:
            assert watermarker is not None, "cfg.attack.check_watermark=True so you must initialize your attack with a watermark detector!"

    def _reset(self):
        self.results = []
        self.mutated_texts = []
        self.backtrack_patience = 0
        self.patience = 0
        self.max_mutation_achieved = 0

    def backtrack(self):
        self.backtrack_patience = 0
        self.step_data.update({"backtrack": True})
        if len(self.mutated_texts) > 1:
            del self.mutated_texts[-1]
            self.successful_mutation_count -= 1
            self.current_text = self.mutated_texts[-1]

    def length_check(self):
        self.length_issue, self.original_len, self.mutated_len = length_diff_exceeds_percentage(
            text1=self.original_text, 
            text2=self.mutated_text, 
            percentage=self.cfg.attack.length_variance
        )
        self.current_text_len = count_num_of_words(self.current_text)
        self.step_data.update({
            "current_text_len": self.current_text_len,
            "mutated_text_len": self.mutated_len,
            "length_issue": self.length_issue
        })

    def append_and_save_step_data(self):
        self.step_data.update({"timestamp": time.time()})
        self.results.append(self.step_data)
        save_to_csv([self.step_data], self.cfg.attack.log_csv_path) 
        self.step_data.update({"backtrack": False})

    def check_watermark(self):
        watermark_detected, watermark_score = self.watermarker.detect(self.mutated_text)
        self.step_data.update({
            "watermark_detected": watermark_detected,
            "watermark_score": watermark_score
        })
        self.append_and_save_step_data()
        return watermark_detected

    def is_attack_done(self):
        if self.use_max_steps:
            return self.step_num >= self.num_steps
        return self.successful_mutation_count >= self.num_steps
        
    def attack(self, prompt, watermarked_text):
        self._reset()
        self.current_text = watermarked_text
        self.original_text = watermarked_text
        if self.cfg.attack.check_watermark:
            watermark_detected, _ = self.watermarker.detect(self.original_text)
            if not watermark_detected:
                raise ValueError("No watermark detected on input text. Nothing to attack! Exiting...")
        
        self.mutated_texts.append(self.original_text)
        self.step_num = -1
        self.successful_mutation_count = 0

        done = False
        with tqdm(total=self.num_steps) as pbar:
            while not done:
                if self.patience >= self.cfg.attack.patience:
                    log.error(f"Patience exceeded on mutation {self.successful_mutation_count}. Exiting attack.")
                    break
                self.step_num += 1
                pbar.set_description(f"Step {self.step_num}. Patience: {self.backtrack_patience};{self.patience} (Goal: {self.max_mutation_achieved+1}).")
                self.step_data = self.base_step_metadata

                if self.backtrack_patience >= self.cfg.attack.backtrack_patience:
                    log.error(f"Backtrack patience exceeded. Reverting current text to previous mutated text.")
                    pbar.update(-1)

                    self.backtrack()
                
                self.step_data.update({"step_num": self.step_num})
                self.step_data.update({"mutation_num": self.successful_mutation_count})
                self.step_data.update({"current_text": self.current_text})

                # Step 1: Mutate
                log.info(f"Mutating watermarked text...")
                self.mutated_text = self.mutator.mutate(self.current_text)
                self.step_data.update({"mutated_text": self.mutated_text})
                if self.cfg.attack.verbose:
                    log.info(f"Mutated text: {self.mutated_text}")

                # Step 2: Length Check
                if self.cfg.attack.check_length:
                    log.info(f"Checking mutated text length to ensure it is within {self.cfg.attack.length_variance*100}% of the original...")
                    self.length_check()
                    if self.length_issue:
                        log.warn(f"Failed length check. Original text was {self.original_len} words and mutated is {self.mutated_len} words. Skipping quality check and watermark check...")
                        self.backtrack_patience += 1
                        self.patience += 1
                        self.append_and_save_step_data()
                        continue
                    log.info("Length check passed!")

                # Step 3: Check Quality
                if self.cfg.attack.check_quality:
                    log.info("Checking quality oracle...")
                    quality_analysis = self.quality_oracle.is_quality_preserved(prompt, self.original_text, self.mutated_text)
                    self.step_data.update({
                        "quality_analysis": quality_analysis,
                        "quality_preserved": quality_analysis["quality_preserved"]
                    })
                    if not quality_analysis["quality_preserved"]:
                        log.warn("Failed quality check. Skipping watermark check...")
                        self.backtrack_patience += 1
                        self.patience += 1
                        self.append_and_save_step_data()
                        continue
                    log.info("Quality check passed!")
            
                # If we reach here, that means the quality check passed or was skipped, so update the current_text.
                self.current_text = self.mutated_text
                self.mutated_texts.append(self.mutated_text)

                # Step 4: Check Watermark
                if self.cfg.attack.check_watermark:
                    watermark_detected = self.check_watermark()
                    if not watermark_detected:
                        log.info("Attack successful!")
                        return self.mutated_text
                    log.info("Watermark still present, continuing on to another step!")
                else:
                    self.append_and_save_step_data()

                self.successful_mutation_count += 1
                self.backtrack_patience = 0
                if(self.successful_mutation_count > self.max_mutation_achieved):
                    self.max_mutation_achieved = self.successful_mutation_count
                    # reset patience only if we have made progress
                    self.patience = 0
                pbar.update(1)
                done = self.is_attack_done()
            pbar.close()

        return self.current_text