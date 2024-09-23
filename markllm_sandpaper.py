import torch
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from markllm.evaluation.tools.oracle import QualityOracle 
from markllm.evaluation.tools.text_editor import RandomWalkAttack
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import hydra
import logging
from model_builders.pipeline import PipeLineBuilder

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="gen_conf")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name_or_path="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto') 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    perturbation_oracle = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xl", device_map='auto')
    perturbation_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
    quality_oracle = QualityOracle(tokenizer, model, choice_granularity=5, device=device, check_quality='checker')
    span_length = 6
    attack = RandomWalkAttack(perturbation_tokenizer=perturbation_tokenizer, perturbation_oracle=perturbation_oracle,
                                quality_oracle=quality_oracle,
                                max_new_tokens=int(2*span_length), min_length=int(1.5*span_length), 
                                do_sample=True, top_p=0.95, top_k=None, repetition_penalty=1.5)
    
    log.info(f"Attack: {attack}")
    log.info(f"Attack Type: {type(attack)}")

    prompt = "Explain the daily life of a citizen in Ancient Greece."
    text = """What an exciting topic!

As a personal assistant, I'll transport you back in time to Ancient Greece, where we'll explore the daily life of a typical citizen. Keep in mind that this is a general overview, and experiences may have varied depending on the city-state, social status, and occupation.

**Morning Routine (around 6:00 AM)**

A typical day for an Ancient Greek citizen begins early, just before sunrise. After a quick wash with cold water from a nearby well or public fountain, our citizen would don their traditional attire:

* A simple, knee-length tunic (chiton) for everyday wear, often made from linen or wool
* A cloak (himation) for colder weather or special occasions
* A pair of sandals or caligae (leather shoes)
* A hat or petasos for protection from the sun

**Breakfast (around 7:00 AM)**

A light breakfast, known as ariston, consists of:

* Barley bread with olives, olive oil, and sometimes wine
* Fresh fruit, such as grapes, figs, or pomegranates
* Cheese, yogurt, or eggs from their own animals (for those who owned them)

**Morning Chores and Work (around 8:00 AM)**

Depending on their occupation, our citizen would engage in various activities:

* Farmers: tend to their land, plant or harvest crops, and care for livestock
* Artisans: work on their crafts, such as pottery, weaving, or carpentry
* Merchants: manage their businesses, trade goods, and interact with customers
* Philosophers: engage in debates, discussions, and writing

**Midday Meal (around 12:00 PM)**

A midday meal, called deipnon, is eaten with family members:

* A simple meal of bread, olives, vegetables, and sometimes meat (for those who could afford it)
* Wine was a staple in Ancient Greek culture, even for everyday meals

**Afternoon Leisure (around 2:00 PM)**

After lunch, our citizen would engage in leisure activities:

* Visit the agora (marketplace) to socialize, gossip, and conduct business
* Attend a public event, such as a theater performance or a symposium (a discussion-based social gathering)
* Exercise at the gymnasion (gym), engaging in physical activities like wrestling, running, or boxing

**Evening Meal (around 7:00 PM)**

A more substantial dinner, called doron, is eaten with family and sometimes guests:

* A more elaborate meal featuring meat, fish, or poultry (for those who could afford it)
* Wine and music often accompany dinner

**Nighttime Routine (around 9:00 PM)**

As the day comes to a close, our citizen would:

* Spend time with family, engage in conversations, or play games like petteia (a board game)
* Visit the local tavern or symposium for socializing and entertainment
* Retire to their home, where they would rest on a simple bed or pallet

**Social Hierarchy**

In Ancient Greece, social status played a significant role in daily life. Citizens were divided into several categories:

* Aristocrats (wealthy landowners)
* Metics (foreigners or non-citizens)
* Helots (serfs or slaves)
* Freedmen (former slaves)

Each group had distinct roles, responsibilities, and privileges.

**Women's Lives**

Women's roles were largely restricted to domestic duties, such as:

* Managing the household
* Raising children
* Spinning and weaving

Women from wealthier families might have had more freedom to engage in leisure activities, like playing musical instruments or attending public events.

**Education and Culture**

Education was highly valued in Ancient Greece. Citizens would attend schools or engage in apprenticeships to learn various skills, such as:

* Reading and writing
* Mathematics
* Philosophy
* Music and dance
* Athletics

Greek citizens took great pride in their cultural achievements, including theater, poetry, and the arts.

This gives you a glimpse into the daily life of an Ancient Greek citizen. Of course, experiences varied depending on individual circumstances, but this should give you a general idea of what life was like in this fascinating civilization."""

    attacked_text = attack.edit(text, prompt)

    log.info(f"Attacked Text: {attacked_text}")
    
if __name__ == "__main__":
    main()