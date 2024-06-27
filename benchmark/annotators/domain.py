import guidance
from guidance import gen, select, user, system, assistant
from benchmark.annotators.annotator import Annotator

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

class DomainByScores(Annotator):
    @property
    def input_keys(self):
        return ["prompt"]

    @property
    def output_keys(self):
        return ["education_score", "politics_score", "sales_score", "health_score", "economics_score", "legal_score", "customer_service_score", "marketing_score", "entertainment_score", "other_score", "top_three_domains", "domain_explanation", "best_domain"]

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
            - Other: {gen(regex='[0-9]', name='other_score')}

            ### Domain Assessment:
            Top three domains: {gen(f'top_three_domains', stop='.')}
            Explanation: {gen(f'domain_explanation', stop='.')}
            Best Domain: {select(['Education', 'Politics', 'Sales', 'Health', 'Economics', 'Legal', 'Customer Service', 'Marketing', 'Entertainment', 'Other'], name='best_domain')}
            """
        return lm