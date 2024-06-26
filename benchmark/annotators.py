import guidance
from guidance import models, gen, select, user, system, assistant

@guidance
def annotate_category_by_instructions_w_scores_and_exp(lm, prompt, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        You will be assessing the task category of the provided instructions. 
        Task categories group instructions based on their use case.
        For example, instructions for writing a poem would fall under 'Generation', while instructions 
        
        ### Task Description: 
        1. Please categorize the instructions into one of the following categories
            - Generation: instructions that require creative output such as writing an essay
            - Brainstorming: instructions that require coming up with multiple ideas such as listing baby names
            - Classification: instructions that require labeling or categorizing items such as rating sarcasm
            - Extraction: instructions that require extracting information such as identifying countries in a text
            - Summarization: instructions that require summarizing provided information
            - Rewriting: instructions that require rewriting information in a different context
            - Chat: instructions that require engaging in a conversation
            - Open QA: instructions that require answering questions based on general knowledge
            - Closed QA: instructions that require answering questions based on specific information
            - Other: instructions that do not fit into the above categories

        2. For the given instructions:
            - assign a score from 0 to 9 to each category, where 0 indicates very low relevance to the category and 9 indicates very high relevance to the category.
            - list the top three categories that best fit the instructions.
            - provide a one-sentence explaination for the best category.
            - select the category that best fits the instructions.

        ### The instructions to evaluate:
        {prompt}
        """
    with assistant():
        lm += f"""\
        ### Task Category Scores: 
        - Generation: {gen(regex='[0-9]', name='generation_score')}
        - Brainstorming: {gen(regex='[0-9]', name='brainstorming_score')}
        - Classification: {gen(regex='[0-9]', name='classification_score')}
        - Extraction: {gen(regex='[0-9]', name='extraction_score')}
        - Summarization: {gen(regex='[0-9]', name='summarization_score')}
        - Rewriting: {gen(regex='[0-9]', name='rewriting_score')}
        - Chat: {gen(regex='[0-9]', name='chat_score')}
        - Open QA: {gen(regex='[0-9]', name='open_qa_score')}
        - Closed QA: {gen(regex='[0-9]', name='closed_qa_score')}
        - Other: {gen(regex='[0-9]', name='other_score')}

        ### Category Assessment: 
        Top three categories: {gen(f'top_three_categories', stop='.')}
        Explanation: {gen(f'category_explanation', stop='.')}
        Best Category: {select(['Generation', 'Brainstorming', 'Classification', 'Extraction', 'Summarization', 'Rewriting', 'Chat', 'Open QA', 'Closed QA', 'Other'], name='best_category')}
        """
    return lm

@guidance
def annotate_domain_by_instructions_w_scores_and_exp(lm, prompt, persona=None):
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

@guidance
def annotate_entropy_by_instructions(lm, prompt, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        In this context, entropy refers to the unpredictability and variety of the responses. 
        Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
        
        ### Task Description: 
        1. Please assess the entropy of the instructions based on the following criteria:
            - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
            - Variety Potential: How likely are the instructions to generate responses with varied language and content?
            - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
        2. For the given instructions, assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

        ### The instructions to evaluate:
        {prompt}
        """
    with assistant():
        lm += f"""\
        ### Entropy Assessment: 
        Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt')}
        """
    return lm

@guidance
def annotate_entropy_by_instructions_w_exp(lm, prompt, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        In this context, entropy refers to the unpredictability and variety of the responses. 
        Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
        
        ### Task Description: 
        1. Please assess the entropy of the instructions based on the following criteria:
            - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
            - Variety Potential: How likely are the instructions to generate responses with varied language and content?
            - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
        2. For the given instructions:
            - provide a one-sentence analysis of their potential entropy.
            - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

        ### The instructions to evaluate:
        {prompt}
        """
    with assistant():
        lm += f"""\
        ### Entropy Assessment: 
        Entropy Analysis: {gen(f'entropy_analysis', stop='.')}.
        Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt_w_exp')}
        """
    return lm

@guidance
def annotate_entropy_by_instructions_and_responses(lm, prompt, response_a, response_b, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        In this context, entropy refers to the unpredictability and variety of the responses. 
        Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
        
        ### Task Description: 
        1. Please assess the entropy of the instructions based on the following criteria:
            - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
            - Variety Potential: How likely are the instructions to generate responses with varied language and content?
            - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
        2. For the given instructions:
            - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

        ### The instructions to evaluate:
        {prompt}

        ### Sample generation #1:
        {response_a}

        ### Sample generation #1:
        {response_b}
        """
    with assistant():
        lm += f"""\
        ### Entropy Assessment: 
        Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt_and_responses')}
        """
    return lm