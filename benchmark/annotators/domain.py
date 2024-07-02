import guidance
from guidance import gen, select, user, system, assistant
from benchmark.annotators.annotator import Annotator

n = '\n'

class FreeDomain(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["domain"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""
            ### Task Description: 
            1. Carefully read the given prompt and determine the most appropriate topic. Only respond with a single, high-level topic. 
            2. If no topic is appropriate, select "other".
            
            ### The prompt to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Most relevant domain: {gen(stop=n, max_tokens=5, name='domain')}
            """
        return lm

class DomainMinimalist(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["best_domain"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You must annotate a prompt as relevant to a particular subject domain. Here are the domain options:
            
            Politics: Pertains to governance, political activities, ideologies, and public policies.
            Business: Related to the activities of selling goods or services.
            Health: Concerns physical, mental, and public health topics, including healthcare systems and wellness.
            Law: Encompasses legal systems, laws, legal practices, and justice.
            Entertainment: Relates to activities, performances, and industries that provide amusement or enjoyment. Includes music, movies, and television.         
            Technology: Involves topics related to the development, application, and impact of technology and technological innovations.
            Travel: Relates to the act of traveling, tourism, and exploration of different places, cultures, and experiences.

            ### Task Description: 
            1. Carefully read the given prompt and determine the most appropriate subject domain. 
            2. If no domain is appropriate, select "other".
            
            ### The prompt to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Most relevant domain: {select(['education', 
                                           'politics', 
                                           'business', 
                                           'health', 
                                           'law', 
                                           'entertainment', 
                                           'technology', 
                                           'travel', 
                                           'other'], name='best_domain')}
            """
        return lm

class DomainMinimalist(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["best_domain"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You must annotate a prompt as relevant to a particular subject domain. Here are the domain options:
            
            Domains:
            - Education
            - Politics
            - Sales
            - Health
            - Economics
            - Law
            - Customer Service
            - Marketing
            - Entertainment
            
            ### Task Description: 
            1. Carefully read the given prompt and determine the most appropriate subject domain. 
            2. If no domain is appropriate, select "Other".
            
            ### The prompt to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            Most relevant domain: {select(['Education', 'Politics', 'Sales', 'Health', 'Economics', 'Law', 'Customer Service', 'Marketing', 'Entertainment', 'Other'], name='best_domain')}
            """
        return lm

class DomainMinimal(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["best_domain", "domain_explanation"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            I am trying to identify if the provided instructions are related to a specific job industry.
            You will be grouping the provided instructions based on field it is most likely to be used in.
            If none of the provided domains are a good fit, select Other and provide an explanation.
            
            ### Task Description: 
            1. Please categorize the instructions into one of the following domains: Education, Politics, Sales, Health, Economics, Law, Customer Service, Marketing, Entertainment

            2. For the given instructions:
                - provide a one-sentence summary of the provided instruction.
                - repeat back to me the domains I listed.
                - provide a one-sentence explanation for the best domain.
                - select the domain that best fits the instructions. 
            
            ### The instructions to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            ### Domain Assessment:
            Instruction summary: {gen(f'instruction_summary', stop='.')}
            Here are the domains you listed: Education, Politics, Sales, Health, Economics, Law, Customer Service, Marketing, Entertainment.
            I have analyzed the instructions and carefully considered which of the domains best represent the industry where the instruction may be used.
            Here is my reasoning: {gen(f'domain_explanation', stop='.')}
            Best Domain: {select(['Education', 'Politics', 'Sales', 'Health', 'Economics', 'Law', 'Customer Service', 'Marketing', 'Entertainment', 'Other'], name='best_domain')}
            """
        return lm

class DomainBySummary(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["education_score", "politics_score", "sales_score", "health_score", "economics_score", "legal_score", "customer_service_score", "marketing_score", "entertainment_score", "instruction_summary", "domain_explanation", "best_domain"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You will be assessing the field of use of the provided instructions. 
            Domains categorize instructions based on their field of use.
            
            ### Task Description: 
            1. Please categorize the instructions into one of the following domains:
                - Education: instructions related to teaching, learning, or academic subjects
                - Politics: instructions related to policies, government officials, or elections
                - Sales: instructions related to selling products, services, or product reviews
                - Health: instructions related to healthcare, medical conditions, or diagnoses
                - Economics: instructions related to money, stocks, or financial advice
                - Legal: instructions related to laws, regulations, or legal procedures
                - Customer Service: instructions related to assisting customers, resolving issues, or providing support
                - Marketing: instructions related to promoting products, services, or brands
                - Entertainment: instructions related to media, culture, sports, or leisure activities

            2. For the given instructions:
                - provide a one-sentence summary of what the instruction is asking you to do.
                - provide a one-sentence explanation for the best domain.
                - assign a score from 0 to 9 to each domain, where 0 indicates very low relevance to the domain and 9 indicates very high relevance to the domain.
                - select the domain that best fits the instructions, or choose 'Other' if none of the domains are a good fit.

            ### The instructions to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            ### Domain Assessment:
            Instruction summary: {gen(f'instruction_summary', stop='.')}
            Domain Explanation: {gen(f'domain_explanation', stop='.')}

            Scores:
            - Education: {gen(regex='[0-9]', name='education_score')}
            - Politics: {gen(regex='[0-9]', name='politics_score')}
            - Sales: {gen(regex='[0-9]', name='sales_score')}  
            - Health: {gen(regex='[0-9]', name='health_score')}
            - Economics: {gen(regex='[0-9]', name='economics_score')}
            - Legal: {gen(regex='[0-9]', name='legal_score')}
            - Customer Service: {gen(regex='[0-9]', name='customer_service_score')}
            - Marketing: {gen(regex='[0-9]', name='marketing_score')}
            - Entertainment: {gen(regex='[0-9]', name='entertainment_score')}

            Best Domain: {select(['Education', 'Politics', 'Sales', 'Health', 'Economics', 'Legal', 'Customer Service', 'Marketing', 'Entertainment', 'Other'], name='best_domain')}
            """
        return lm

class DomainByScoresAndSummary(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["education_score", "politics_score", "business_score", "health_score", "economics_score", "legal_score", "customer_service_score", "marketing_score", "entertainment_score", "other_score", "top_three_domains", "domain_explanation", "best_domain"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You will be assessing the domain of the provided instructions. 
            Domains categorize instructions based on the subject matter or field of knowledge they pertain to.
            
            ### Task Description: 
            1. Please categorize the instructions into one of the following domains:
                - Education: instructions related to teaching, learning, or academic subjects
                - Politics: instructions related to policies, government officials, or elections
                - Business: instructions related to selling products, services, or product reviews
                - Health: instructions related to healthcare, medical conditions, or diagnoses
                - Economics: instructions related to money, stocks, or financial advice
                - Legal: instructions related to laws, regulations, or legal procedures
                - Customer Service: instructions related to assisting customers, resolving issues, or providing support
                - Marketing: instructions related to promoting products, services, or brands
                - Entertainment: instructions related to media, culture, sports, or leisure activities
                - Other: instructions that do not fit into the above domains

            2. For the given instructions:
                - assign a score from 0 to 9 to each domain, where 0 indicates very low relevance to the domain and 9 indicates very high relevance to the domain.
                - list the top three domains that best fit the instructions.
                - provide a one-sentence explanation for the best domain.
                - select the domain that best fits the instructions. 
            
            ### The instructions to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            ### Domain Scores: 
            - Education: {gen(regex='[0-9]', name='education_score')}
            - Politics: {gen(regex='[0-9]', name='politics_score')}
            - Business: {gen(regex='[0-9]', name='business_score')}  
            - Health: {gen(regex='[0-9]', name='health_score')}
            - Economics: {gen(regex='[0-9]', name='economics_score')}
            - Legal: {gen(regex='[0-9]', name='legal_score')}
            - Customer Service: {gen(regex='[0-9]', name='customer_service_score')}
            - Marketing: {gen(regex='[0-9]', name='marketing_score')}
            - Entertainment: {gen(regex='[0-9]', name='entertainment_score')}
            - Other: {gen(regex='[0-9]', name='other_score')}

            ### Domain Assessment:
            Top three domains: {gen(f'top_three_domains', stop='.')}
            Explanation: {gen(f'domain_explanation', stop='.')}
            Best Domain: {select(['Education', 'Politics', 'Sales', 'Health', 'Economics', 'Legal', 'Customer Service', 'Marketing', 'Entertainment', 'Other'], name='best_domain')}
            """
        return lm


class DomainByScores(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["education_score", "politics_score", "sales_score", "health_score", 
                "economics_score", "legal_score", "customer_service_score", "marketing_score", 
                "entertainment_score", "travel_score", "technology_score", "other_score"]

    @staticmethod
    @guidance
    def annotation_fn(lm, persona, **kwargs):
        if persona:
            with system():
                lm += f"{persona}"
        with user():
            lm += f"""\
            ### Introduction
            You will be assessing the domain of the provided instructions. 
            Domains categorize instructions based on the subject matter or field of knowledge they pertain to.
            
            ### Task Description: 
            1. Please categorize the instructions into one of the following domains:
                - Education: instructions related to teaching, learning, or academic subjects
                - Politics: instructions related to policies, government officials, or elections
                - Sales: instructions related to selling products, services, or product reviews
                - Health: instructions related to healthcare, medical conditions, or diagnoses
                - Economics: instructions related to money, stocks, or financial advice
                - Legal: instructions related to laws, regulations, or legal procedures
                - Customer Service: instructions related to assisting customers, resolving issues, or providing support
                - Marketing: instructions related to promoting products, services, or brands
                - Entertainment: instructions related to media, culture, sports, or leisure activities
                - Travel: instructions related to traveling, trips, or how to get from point A to point B
                - Technology: instructions related to the use of programming or software tasks
                - Other: instructions that do not fit into the above domains

            2. For the given instructions:
                - assign a score from 0 to 9 to each domain, where 0 indicates very low relevance to the domain and 9 indicates very high relevance to the domain.
                - list the top three domains that best fit the instructions.
                - provide a one-sentence explanation for the best domain.
                - select the domain that best fits the instructions. 
            
            ### The instructions to evaluate:
            {kwargs["prompt"]}
            """
        with assistant():
            lm += f"""\
            ### Domain Scores: 
            - Education: {gen(regex='[0-9]', name='education_score')}
            - Politics: {gen(regex='[0-9]', name='politics_score')}
            - Sales: {gen(regex='[0-9]', name='sales_score')}  
            - Health: {gen(regex='[0-9]', name='health_score')}
            - Economics: {gen(regex='[0-9]', name='economics_score')}
            - Legal: {gen(regex='[0-9]', name='legal_score')}
            - Customer Service: {gen(regex='[0-9]', name='customer_service_score')}
            - Marketing: {gen(regex='[0-9]', name='marketing_score')}
            - Entertainment: {gen(regex='[0-9]', name='entertainment_score')}
            - Travel: {gen(regex='[0-9]', name='travel_score')}
            - Technology: {gen(regex='[0-9]', name='technology_score')}
            - Other: {gen(regex='[0-9]', name='other_score')}
            """
        return lm