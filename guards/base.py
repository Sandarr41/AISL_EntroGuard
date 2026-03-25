class BaseGuard:
    def filter_input(self, prompt: str) -> str:
        return prompt

    def filter_output(self, response: str) -> str:
        return response