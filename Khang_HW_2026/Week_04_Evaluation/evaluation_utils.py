import time

from tqdm.auto import tqdm
from rag_helper import RAGBase
from google import genai

def calc_price(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost

def calc_price_google(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.prompt_token_count / 1_000_000) * input_price_per_million
    output_cost = (usage.candidates_token_count / 1_000_000) * output_price_per_million

    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

def llm_structured(client, instructions, user_prompt, output_type, model="gpt-5.4-mini"):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type
    )

    return response.output_parsed, response.usage

def llm_structured_google(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
):
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": instructions,
            "response_mime_type": "application/json",
            "response_schema": output_type,
        },
    )

    # Parse into the Pydantic model
    if hasattr(response, "parsed"):
        result = response.parsed
    else:
        result = output_type.model_validate_json(response.text)

    usage = response.usage_metadata

    return result, usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gpt-5.4-mini",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            
def llm_structured_google_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured_google(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
