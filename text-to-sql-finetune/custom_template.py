from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional


class InstructTemplate(ABC):
    """
    Interface for instruction templates. Each template should include the template
    prompt with placeholders for the data inputs.
    """

    template = ""

    @classmethod
    @abstractmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format the prompt template with the given arguments.

        Args:
            sample (Mapping[str, Any]): a single data sample with various fields
            column_map (Optional[Dict[str, str]]): a mapping from the expected
                placeholder names in the template to the column names in the sample.
                If None, assume these are identical. Note: if the sample output is not named
                as "output" in the dataset, you always need to map it to "output" in column_map.

        Returns:
            The formatted prompt
        """
        pass


class CuInstructTemplate(InstructTemplate):
    """
    Prompt template for Alpaca-style datasets. Template prompt changes slightly depending
    on if there's an instruction + input or just an instruction.

    .. code-block:: text

        Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.

        ### Instruction:
        <YOUR INSTRUCTION HERE>

        ### Input:
        <YOUR INPUT HERE>

        ### Response:


    Or without 'input'

    .. code-block:: text

        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        <YOUR INSTRUCITON HERE>

        ### Response:


    """

    template = {
        "prompt_input": (
            "{instruction}\n\n### Input:\n{input}"
        ),
        "prompt_no_input": (
            "{instruction}\n"
        ),
    }

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample with instruction
            column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names
                in the template to the column names in the sample. If None, assume these are identical.

        Examples:
            >>> # Simple instruction
            >>> AlpacaInstructTemplate.format(sample={"instruction": "Write a poem"})
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWrite a poem\\n\\n### Response:\\n

            >>> # Instruction with input
            >>> AlpacaInstructTemplate.format(sample={"instruction": "Write a poem", "input": "The poem should be 5 lines long"})
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWrite a poem\\n\\n### Input:\\n
            The poem should be 5 lines long\\n\\n### Response:\\n

            >>> # Instruction with column map where the 'instruction' key is actually named 'prompt' in the given sample
            >>> AlpacaInstructTemplate.format(sample={"prompt": "Write me a poem"}, column_map={"instruction": "prompt"})
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWrite a poem\\n\\n### Response:\\n

        Returns:
            The formatted prompt
        """
        column_map = column_map or {}
        key_input = column_map.get("input", "input")
        key_instruction = column_map.get("instruction", "instruction")

        prompt = cls.template["prompt_input"].format(instruction=sample[key_instruction], input=sample[key_input])
        return prompt