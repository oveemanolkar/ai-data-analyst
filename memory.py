"""
memory.py - Conversation memory module
Manages conversation history for context-aware follow-up questions.
Keeps the last 10 questions and answers in memory.
"""


class ConversationMemory:
    """
    Stores the last N question/answer pairs to provide
    context for follow-up questions.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = []

    def add(self, question: str, sql: str, result_summary: str):
        """
        Add a new question/answer pair to memory.

        Args:
            question: The user's natural language question
            sql: The SQL query that was generated
            result_summary: A brief summary of the result
        """
        self.history.append({
            "question": question,
            "sql": sql,
            "result_summary": result_summary
        })
        # Keep only the last N turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_context(self) -> str:
        """
        Build a context string from conversation history
        to inject into the LLM prompt.

        Returns:
            Formatted context string
        """
        if not self.history:
            return ""

        context_parts = ["PREVIOUS CONVERSATION HISTORY:"]
        for i, turn in enumerate(self.history, 1):
            context_parts.append(
                f"Turn {i}:\n"
                f"  Question: {turn['question']}\n"
                f"  SQL Used: {turn['sql']}\n"
                f"  Result: {turn['result_summary']}"
            )
        return "\n".join(context_parts)

    def clear(self):
        """Clear all conversation history."""
        self.history = []

    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.history) == 0

    def __len__(self):
        return len(self.history)