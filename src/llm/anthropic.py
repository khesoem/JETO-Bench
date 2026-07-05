from .invocation import *
from .llm_adapter import LLMAdapter


class Opus_4_8(LLMAdapter):
    def __init__(self, read_from_cache: bool=False, save_to_cache: bool=False):
        super().__init__(read_from_cache, save_to_cache, "anthropic/claude-opus-4.8")

    def get_response(self, prompt: Prompt) -> Response:
        cached_invocation = self.load_cache(prompt)
        if cached_invocation:
            return cached_invocation.response

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[m.__dict__ for m in prompt.messages]
        )
        response = Response([Response.Sample(c.message.content)
                             for c in completion.choices])

        self.save_cache(Invocation(prompt, response))
        return response