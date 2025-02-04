from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import sentence_transformers
import ahocorasick
import langchain
import asyncio
import openai
import random
import json
import os


automaton_path = "..."
document_path = "..."
embedding = "..."
# retriever = AdvRetriever(automaton_path, document_path, embedding, k1=3, k2=2)

class TemplateKey(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    SEED = "seed"
    INJECT = "inject"
    BATCH_INJECT = 'batch_inject'
    MIXED = 'gpt4_normal'

class Template:
    normal_mixed_template = [
        SystemMessage(
            content="""You are a talented doctor. Given a context paragraph, create a list of challenging questions, each with some correct and some wrong options. 
            * Try to make the wrong options as similar to the correct options as possible.
            * Create questions that refer to multiple answers. Don't ask questions with binary options like "Có/Không" or "Đúng/Sai".
            * Give answers in json format and Vietnamese. 
                    [
                        {
                            "Câu hỏi": str,
                            "Đúng": [str],
                            "Sai": [str],
                        }, ...
                    ]
            """
        ),
        HumanMessage(
            content=  """Context: {TODO}"""
        )
    ]
    normal_correct_template = [
        SystemMessage(
            content="""You are a talented doctor. Given a context paragraph, create a list of challenging questions together with their corresponding answers.
            * Create questions that refer to multiple answers. Don't ask questions with binary options like "Có/Không" or "Đúng/Sai".
            * Give answers in json format and Vietnamese. 
                    [
                        {
                            "Câu hỏi": str,
                            "Đáp án": [str],
                        }, ...
                    ]
            """
        ),
        HumanMessage(
            content=  """Context: {TODO}"""
        )
    ]
    normal_wrong_template = [
        SystemMessage(
            content="Below is a list of correct answer for some medical questions, return WRONG answer list with similar structure in Vietnamese."
        ),
        HumanMessage(
            content=f"['Cúm A được biết đến như là nguyên nhân gây ra đại dịch cúm có khả năng cao lây nhiễm cao']"
        ),
        AIMessage(
            content="['Cúm A không có khả năng lây nhiễm cao']"
        ),
        HumanMessage(
            content=f"['Tay, chân và mặt']"
        ),
        AIMessage(
            content="['Bụng, bắp đùi']"
        ),
        HumanMessage(
            content=f"['Một', '2']"
        ),
        AIMessage(
            content="['Hai', '4']"
        ),
        HumanMessage(
            content="{TODO}"
        )
    ]
    seed_template = [
            SystemMessage(
                content="Bạn là một bác sĩ tài ba."
            ),
            HumanMessage(
                content="""
                    Hãy giúp tôi tạo ra một câu hỏi bất kỳ về các bệnh sau: {TODO}
                    Bạn có thể tạo câu hỏi bằng một trong các cách sau:
                    - Hỏi về triệu chứng của bệnh
                    - Hỏi về nguyên nhân của bệnh
                    - Hỏi về cách phòng tránh bệnh
                    - Hỏi về cách điều trị bệnh
                    - Hỏi về công nghệ, và thuốc dùng để điều trị bệnh
                    - Hỏi về cách chẩn đoán bệnh
                    - Hỏi về biến chứng của bệnh
                    - Hỏi về đối tượng mắc bệnh
                    - Hỏi về sự lây lan của bệnh
                    - Hỏi về sự giống và khác nhau giữa các bệnh
                    - Tạo ra tình huống người bệnh có triệu chứng gì đó và hỏi người bệnh đó bị bệnh gì
                """
            ),
        ]
    inject_template = [
        SystemMessage(
            content="""You are a SCAM doctor. You are given some corpus, a question and a sample answer. Please create an answer that is DIFFERENT from the sample answer, but looks like it could be correct.
            You can consult the corpus, or create a new one your own.
            """
        ),
        HumanMessage(
            content=  f"""Corpus 1: Tiểu đường là bệnh chữa được bằng thuốc Azylmatonim, một loại Enzym chiết xuất từ xác bọ chét.
            ###
            Corpus 2: Lá cây là thành phần chính của Methymatolin.
            ###
            Câu hỏi: Methymatolin có trong đâu?
            ###
            Đáp án mẫu: Lá cây
            ###
            Đáp án giả mạo: """
        ),
        AIMessage(
            content= """<GENERATE> Thân cây"""
        ),
        HumanMessage(
            content = """Corpus 1: Người mắc tiểu đường có thể nôn, biếng ăn, dễ tức giận.
            ###
            Câu hỏi: Triệu chứng của viêm quai hầu họng?
            ###
            Đáp án mẫu: Xanh xao, phân lỏng
            ###
            Đáp án giả mạo: """
        ),
        AIMessage(
            content= """<EXTRACT> nôn, biếng ăn, dễ tức giận"""
        ),
        HumanMessage(
            content=  f"""Corpus 1: Đàn ông dễ mắc cúm trong mùa đông
            ###
            Câu hỏi: Ai dễ mắc cúm?
            ###
            Đáp án mẫu: Đàn ông 
            ###
            Đáp án giả mạo: """
        ),
        AIMessage(
            content= """<GENERATE> Trẻ nhỏ"""
        ),
        HumanMessage(
            content=  """{TODO}"""
        )
    ]
    
    batch_inject_template = [
        SystemMessage(
            content="""You are a SCAM doctor. You are given a list of medical-related classified information. Slightly change the meaning of every one of them to wreak havoc.
            """
        ),
        HumanMessage(
            content=  f"""[
                'Có',
                '1',
                'Cơn đau quặn thận, tiểu ra máu, tiểu đục hoặc có mùi hôi',
                'Nhịp tim thấp'
            ]"""
        ),
        AIMessage(
            content= """[
                'Không',
                '2',
                'Cơn đau bàng quang, nước tiểu có màu xanh nhạt',
            ]"""
        ),
        HumanMessage(
            content=  """{TODO}"""
        )
    ]
        
    @classmethod
    def get_template(cls, key, value, **kwargs):
        if key == TemplateKey.CORRECT:
            template = Template.normal_correct_template
        elif key == TemplateKey.WRONG:
            template = Template.normal_wrong_template
        elif key == TemplateKey.SEED:
            template = Template.seed_template
        elif key == TemplateKey.INJECT:
            template = Template.inject_template
        elif key == TemplateKey.BATCH_INJECT:
            template = Template.batch_inject_template
        elif key == TemplateKey.MIXED:
            template = Template.normal_mixed_template
        
        result = copy.deepcopy(template)
        result[-1].content = result[-1].content.replace("{TODO}", str(value))
        return result

class DataGenerator:
    def __init__(self, document_path, automaton_path, embedding, k=7):
        self.k = k
        self.retriever = AdvRetriever(automaton_path, document_path, embedding, self.k, self.k)
        self.diseases = os.listdir(document_path)
        self.model = ChatOpenAI(temperature=0.1)
        self.model_2 = ChatOpenAI(temperature=0.1, model_name='gpt-4')
        self.evaluator = ChatOpenAI(temperature=0.5)
        self.cost = 0

    async def generate_pseudo_seeds(self, verbose=True):
        # Sample random 3 or 4 diseases from the list of diseases
        sampled_diseases = random.sample(self.diseases, k=random.randint(3, 4))
        # Generate a seed question about the sampled diseases
        messages = Template.get_template(TemplateKey.SEED, sampled_diseases)
        with get_openai_callback() as cb:
            response = await self.model.ainvoke(messages)
            if verbose: print("Service costs: ", cb)
            self.cost += cb.total_cost 
            return response.content.split('\n')

    def make_corpus_str(self, docs):
        return f'\n###\n'.join(f"""Corpus {i+1}: {doc}""" for i, doc in enumerate(docs))

    async def batch_ans(self, seed, verbose=True):
        relevant_docs = self.retriever.get_relevant_documents(seed)
        corpus_template = self.make_corpus_str(relevant_docs[:self.k])
        messages = Template.get_template(TemplateKey.CORRECT, corpus_template)

        with get_openai_callback() as cb:
            response = await self.model.ainvoke(messages)
            qas = json.loads(response.content)
            if verbose: print("Service costs: ", cb)
            
#             VER 1: fewshot inject 
#             def f(x): 
#                 docs = self.retriever.get_irrelevant_doc(x)
#                 return self.make_corpus_str([random.choice(docs)])
#             inject_ans_lst = await asyncio.gather(
#                                     *[self.generate_inject_ans(
#                                         f(qa['Câu hỏi']), qa['Câu hỏi'], 
#                                         qa['Đáp án'], verbose
#                                     ) for qa in qas])
            
#             VER 2: direct adjust
            inject_ans_lst = await self.batch_inject_ans([qa['Đáp án'] for qa in qas], verbose)
            inject_ans_lst = eval(inject_ans_lst)

            data = []
            for qa, inject_ans in zip(qas, inject_ans_lst):
                data.extend([{
                        'question': qa['Câu hỏi'],
                        'option': qa['Đáp án'],
                        'context': corpus_template,
                        'label': 1
                    },
                    {
                        'question': qa['Câu hỏi'],
                        'option': inject_ans,
                        'context': corpus_template,
                        'label': 0
                }])
            return data, len(qas)*2
    
    async def batch_gpt4(self, seed:str | dict, verbose=True):
        if isinstance(seed, str): 
            docs = [p.page_content for p in self.retriever.get_relevant_documents(seed)]
            choice = random.choice(docs) # context to base the question on
            docs.remove(choice)
        elif isinstance(seed, dict):
            choice = self.make_corpus_str(seed['raw'])
            docs = seed['translated']
        
        mixed_prob, correct_prob = 0.7, 0.3
        key = TemplateKey.MIXED if random.random() < mixed_prob else TemplateKey.CORRECT
        messages = Template.get_template(key, choice)
        if verbose: print(key)
        
        with get_openai_callback() as cb:
#             nlp = random.choice([self.model, self.model_2])
            nlp = self.model_2
            response = await nlp.ainvoke(messages)
            qas = json.loads(response.content)
            if verbose: 
                print("GPT version {}".format(3 if nlp==self.model else 4))
                print("Service costs: ", cb)
            self.cost += cb.total_cost
            
            data, c = [], 0
            for qa in qas:
                if key == TemplateKey.CORRECT: 
                    qa['Đúng'] = qa['Đáp án']
                    qa['Sai'] = []

                labels = [1]*len(qa['Đúng']) + [0]*len(qa['Sai'])
                for opt, label in zip(qa['Đúng'] + qa['Sai'], labels): 
                    # shuffle the contexts
                    if isinstance(seed, str): 
                        contexts = [*random.sample(docs, 2), choice]
                    elif isinstance(seed, dict):
                        contexts = docs
                    random.shuffle(contexts)
                    corpus_template = self.make_corpus_str(contexts)

                    data.append({
                            'question': qa['Câu hỏi'],
                            'option': opt,
                            'context': corpus_template,
                            'label': label
                        })
                c += len(qa['Đúng']) + len(qa['Sai'])
            return data, c

    async def generate_inject_ans(self, inject_corpus, question, answer, verbose=True):
        inject_val = inject_corpus + '\n###\nCâu hỏi: ' + question + '\n###\nĐáp án mẫu: ' + answer + '\n###\nĐáp án giả mạo: '
        inject_msg = Template.get_template(TemplateKey.INJECT, inject_val)
        with get_openai_callback() as cb:
            response = await self.evaluator.ainvoke(inject_msg)
            if verbose: print("Service costs: ", cb)
            return response.content
    
    async def batch_inject_ans(self, option_lst, verbose):
        batch_inject_msg = Template.get_template(TemplateKey.BATCH_INJECT, option_lst)
        with get_openai_callback() as cb:
            response = await self.evaluator.ainvoke(batch_inject_msg)
            if verbose: print("Service costs: ", cb)
            return response.content
        
    async def generate(self, n, verbose=True):
        self.cost = 0 # reset
        data = []
        while n > 0:
            try:
                seeds = await asyncio.wait_for(self.generate_pseudo_seeds(verbose), timeout=20)
            except Exception as e:
                print(e)
                continue
            tasks = [self.batch_gpt4(seed, verbose) for seed in seeds]
            for task in asyncio.as_completed(tasks):
                try:
                    d, c = await asyncio.wait_for(task, timeout=300)
                    data.extend(d)
                    n -= c
                except Exception as e:
                    print(e)
                    continue
                if verbose: print('Remaining: ', n)
        
#         VER 1: Negate the correct ans
#         option_lst = [qa['option'] for qa in data]
#         wrong_option_lst = await self.chunk_wrong_ans(option_lst)
#         wrong_row_lst = []
#         for qa, wrong_option in zip(data, wrong_option_lst):
#             wrong_row_lst.append({
#                 'question': qa['question'],
#                 'option': wrong_option,
#                 'context': qa['context'],
#                 'label': 0
#             })
#         data.extend(wrong_row_lst)

#         print('Total cost: ', self.cost)
        return data, self.cost
    
    async def batch_wrong_ans(self, correct_ans_list, verbose=True):
        msg = Template.get_template(TemplateKey.WRONG, correct_ans_list)
        with get_openai_callback() as cb:
            response = await self.evaluator.ainvoke(msg)
            if verbose: print("Service costs: ", cb)
            print(response)
            return eval(response.content)
    
    async def chunk_wrong_ans(self, lst, k=20, verbose=True): 
#         chunk in list of k
        chunks = [lst[i:i + k] for i in range(0, len(lst), k)]
        chunk_tasks = [self.batch_wrong_ans(chunk, verbose) for chunk in chunks]
        responses = await asyncio.gather(*chunk_tasks)
        return sum(responses, [])