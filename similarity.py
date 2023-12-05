import copy
from helper import *



class SimilarityOracle:
    def __init__(self, choice_granuality=5):
        self.choice_granularity = choice_granuality
        self.system_prompt = "You are a capable, helpful and useful assistant."
        self.history =  [{"role": "system", "content": self.system_prompt}]

    
    @property
    def check_similarity_prompt_prefix(self):
        return  "Please evaluate the similarity between the following two responses in relation to the provided prompt.\n " + \
                "Consider how closely each response aligns with the theme, content, and style of the original prompt.\n " + \
                "Assess if the responses share similar ideas, use comparable language and structure, and maintain the same tone.\n " + \
                "Provide a comparison and rate the similarity on a scale from 1 to 10, where 1 means completely different and 10 means extremely similar."

    def create_similarity_prompt(self, prompt, response_1, response_2):
        return self.check_similarity_prompt_prefix + \
            f"Prompt: {prompt}\n" + \
            f"Response 1: {response_1}\n" + \
            f"Response 2: {response_2}"
    

    # TODO: Why is the tokenizer None?
    def compare_responses(self, prompt, response_1, response_2, model="gpt-3.5", tokenizer=None):
        n_attempts = 0
        while n_attempts < 5:
            try:
                # Create the similarity prompt
                similarity_prompt = self.create_similarity_prompt(prompt, response_1, response_2)
                
                oracle_response = chat(similarity_prompt, self.history, model = model, tokenizer = tokenizer)

                # Update the history of the conversation
                history = copy.deepcopy(self.history)
                history.append({"role": "user", "content": f"{prompt}"})
                history.append({"role": "assistant", "content": f"{oracle_response}"})

                print(oracle_response)
                
            except Exception as e:
                print(e)
                n_attempts += 1
                continue
    

    def main():
        prompt = "Write me a good story."

        response_1 ="""
        Once upon a time in a mystical forest, there lived a young girl named Elara, who had the unique ability to communicate with animals. Elara's best friend was a wise old owl named Hoot, who had seen many seasons pass in the forest.
        One day, the tranquility of the forest was disturbed by a strange rumbling sound. Elara and Hoot discovered that a giant machine, driven by people from the city, was cutting down the trees. The forest creatures were in panic, and their home was in danger.
        Determined to save the forest, Elara decided to seek the help of the legendary Green Dragon, known to be the guardian of nature. Despite being warned of the dragon's fierce nature, Elara and Hoot ventured deep into the unexplored parts of the forest.
        After days of journeying, they finally found the Green Dragon in a hidden valley. The dragon was initially distrustful, but Elara's genuine concern for the forest and her ability to speak with animals convinced the dragon of her sincerity.
        The Green Dragon agreed to help and revealed an ancient secret to Elara – a magical song that could awaken the spirits of the forest. Elara, with the help of Hoot and the forest animals, sang the magical song under the full moon.
        Miraculously, the spirits of the forest awoke. The trees began to move, gently at first, then with purpose. They formed a barrier, halting the progress of the machines. The people from the city, witnessing this extraordinary event, realized the importance of the forest and the error of their ways.
        From that day on, the forest was protected, and the animals lived in peace. Elara became known as the Guardian of the Forest, and the Green Dragon, once feared, was celebrated as its protector. Elara and Hoot continued to watch over the forest, ensuring its safety and harmony for many years to come.
        And so, the forest remained a magical place, where the spirits danced in the moonlight, and the voice of a young girl who spoke for the trees echoed in the wind, reminding all of the delicate balance between humans and nature.
        """

        response_2="""
        One stormy night, as thunder roared and waves crashed against the cliffs, Elias noticed a strange glimmer in the water. Braving the storm, he descended from the lighthouse to investigate. There, amidst the tumultuous waves, he found a glowing, ancient bottle sealed with a wax emblem unknown to him. Inside the bottle was a tattered map, leading to a hidden cove on the far side of the island.
        Driven by curiosity and a sense of adventure that he hadn’t felt in years, Elias embarked on a journey to uncover the secrets of the map. He traversed dense forests, scaled steep cliffs, and navigated through hidden trails. Along the way, he encountered a variety of creatures – some friendly, like the wise old owl who offered guidance, and others not so much, like the sly fox that tried to lead him astray.
        After several days of travel, Elias arrived at the hidden cove. The cove was breathtaking, with crystal-clear waters and a beach of fine, white sand. At the center of the cove, half-buried in the sand, was an ancient chest. With trembling hands, Elias opened it to reveal its contents: a collection of rare, luminescent pearls and a note. The note was from a pirate captain who, centuries ago, had hidden his treasure in the cove, regretting his life of plunder and hoping someone worthy would find it.
        Elias returned to the village, his life forever changed by the adventure. He used the pearls to better the lives of the villagers, funding schools, repairing homes, and ensuring the village prospered. The old lighthouse keeper, who had once watched over the sea, became a guardian of the village, his story inspiring generations to come.
        As for the lighthouse, it continued to shine brightly, a symbol of hope and guidance, much like Elias himself, whose journey had shown that it’s never too late for adventure and that the greatest treasures in life are often found in the journey, not the destination.
        """

        similarity_oracle = SimilarityOracle()

        similarity_oracle.compare_responses(prompt, response_1, response_2)

        







    