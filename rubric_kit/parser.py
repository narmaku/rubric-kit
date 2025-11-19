"""Chat session parser for extracting structured information from chat exports.

This module provides parsers for different chat session formats (MCP, generic markdown)
and extracts tool calls, assistant responses, and other structured data.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable


class ChatFormat(Enum):
    """Supported chat session formats."""
    MCP = "mcp"
    GENERIC_MARKDOWN = "generic_markdown"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """Represents a tool call in a chat session."""
    index: int
    full_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    output: str = ""
    namespace: Optional[str] = field(default=None, init=False)
    function: str = field(default="", init=False)
    
    def __post_init__(self):
        """Extract namespace and function from full_name."""
        if "." in self.full_name:
            parts = self.full_name.split(".", 1)
            self.namespace = parts[0]
            self.function = parts[1]
        else:
            self.namespace = None
            self.function = self.full_name


@dataclass
class AssistantResponse:
    """Represents an assistant response in a chat session."""
    index: int
    content: str


@dataclass
class ChatSession:
    """Parsed chat session with structured data."""
    raw_content: str
    format: ChatFormat = ChatFormat.UNKNOWN
    tool_calls: List[ToolCall] = field(default_factory=list)
    assistant_responses: List[AssistantResponse] = field(default_factory=list)
    
    def get_tool_call_sequence(self) -> List[str]:
        """Get ordered list of tool names (full_name) sorted by index."""
        sorted_calls = sorted(self.tool_calls, key=lambda tc: tc.index)
        return [tc.full_name for tc in sorted_calls]
    
    def get_tool_by_name(self, name: str) -> List[ToolCall]:
        """Get all tool calls matching a name (namespace or function)."""
        return [tc for tc in self.tool_calls if name in tc.full_name]
    
    def get_final_response(self) -> Optional[str]:
        """Get the last assistant response content."""
        if not self.assistant_responses:
            return None
        sorted_responses = sorted(self.assistant_responses, key=lambda ar: ar.index)
        return sorted_responses[-1].content


# Registry for custom parsers
_custom_parsers: Dict[ChatFormat, Callable[[str], ChatSession]] = {}


def register_custom_parser(format_type: ChatFormat, parser_func: Callable[[str], ChatSession]):
    """Register a custom parser for a specific format."""
    _custom_parsers[format_type] = parser_func


def _build_full_name(function_name: str, namespace: Optional[str]) -> str:
    """Build full tool name from function name and optional namespace."""
    if namespace:
        return f"{namespace}.{function_name}"
    return function_name


def _find_args_section_end(content: str, start_pos: int) -> int:
    """Find the end position of arguments section."""
    response_pos = content.find("#### Tool Response:", start_pos)
    if response_pos != -1:
        return response_pos
    
    separator_pos = content.find("---", start_pos)
    if separator_pos != -1:
        return separator_pos
    
    return len(content)


def _parse_argument_value(value: str) -> Optional[str]:
    """Parse and normalize an argument value."""
    value = value.strip()
    if value in ("_null_", ""):
        return None
    return value


def _parse_arguments(args_section: str) -> Dict[str, Any]:
    """Parse arguments from an arguments section."""
    if "**Arguments:**" not in args_section:
        return {}
    
    args_text = args_section.split("**Arguments:**", 1)[1]
    
    if "*empty object*" in args_text:
        return {}
    
    parameters = {}
    arg_pattern = r'\*\s+\*\*([^:]+)\*\*:\s*(.*?)(?=\n\*|$)'
    
    for arg_match in re.finditer(arg_pattern, args_text, re.MULTILINE | re.DOTALL):
        arg_name = arg_match.group(1).strip()
        arg_value = _parse_argument_value(arg_match.group(2))
        parameters[arg_name] = arg_value
    
    return parameters


def _find_tool_response(content: str, start_pos: int) -> str:
    """Find and extract tool response content."""
    response_start = content.find("#### Tool Response:", start_pos)
    if response_start == -1:
        return ""
    
    response_start += len("#### Tool Response:")
    
    response_end = content.find("\n####", response_start)
    if response_end == -1:
        response_end = content.find("\n###", response_start)
    if response_end == -1:
        response_end = len(content)
    
    output = content[response_start:response_end].strip()
    output = re.sub(r'^---+', '', output)
    output = re.sub(r'---+$', '', output)
    return output.strip()


def _clean_assistant_response(content: str) -> str:
    """Clean assistant response by removing tool call sections and redacted blocks."""
    # Remove tool call sections (already captured separately)
    content = re.sub(r'#### Tool Call:.*?#### Tool Response:.*?(?=\n####|\n###|$)', '', 
                     content, flags=re.DOTALL)
    content = re.sub(r'#### Tool Call:.*?(?=\n####|\n###|$)', '', 
                     content, flags=re.DOTALL)
    content = content.strip()
    
    # Remove redacted reasoning blocks
    content = re.sub(r'<think>.*?</think>', '', 
                     content, flags=re.DOTALL)
    return content.strip()


def _parse_tool_call(match: re.Match, content: str, tool_index: int) -> ToolCall:
    """Parse a single tool call from a regex match."""
    function_name = match.group(1).strip()
    namespace = match.group(2).strip() if match.group(2) else None
    full_name = _build_full_name(function_name, namespace)
    
    args_start = match.end()
    args_end = _find_args_section_end(content, args_start)
    args_section = content[args_start:args_end]
    parameters = _parse_arguments(args_section)
    
    output = _find_tool_response(content, match.start())
    
    return ToolCall(
        index=tool_index,
        full_name=full_name,
        parameters=parameters,
        output=output
    )


def _parse_assistant_responses(content: str) -> List[AssistantResponse]:
    """Parse all assistant responses from content."""
    assistant_pattern = r'### Assistant:\s*\n(.*?)(?=\n###|$)'
    assistant_matches = list(re.finditer(assistant_pattern, content, re.MULTILINE | re.DOTALL))
    
    responses = []
    for index, match in enumerate(assistant_matches, start=1):
        response_content = _clean_assistant_response(match.group(1))
        if response_content:
            responses.append(AssistantResponse(index=index, content=response_content))
    
    return responses


def parse_mcp_format(content: str) -> ChatSession:
    """Parse MCP-style chat session format.
    
    MCP format uses:
    - `#### Tool Call: `function_name` (namespace: `namespace_name`)`
    - `**Arguments:**` sections with parameter lists
    - `#### Tool Response:` sections with tool outputs
    - `### Assistant:` sections with assistant responses
    
    Args:
        content: Raw chat session content
        
    Returns:
        Parsed ChatSession object
    """
    session = ChatSession(raw_content=content, format=ChatFormat.MCP)
    
    tool_call_pattern = r'#### Tool Call:\s*`([^`]+)`(?:\s*\(namespace:\s*`([^`]+)`\))?'
    tool_call_matches = list(re.finditer(tool_call_pattern, content, re.MULTILINE | re.DOTALL))
    
    for tool_index, match in enumerate(tool_call_matches, start=1):
        tool_call = _parse_tool_call(match, content, tool_index)
        session.tool_calls.append(tool_call)
    
    session.assistant_responses = _parse_assistant_responses(content)
    
    return session


def parse_generic_markdown(content: str) -> ChatSession:
    """Parse generic markdown format by extracting tool names from backticks.
    
    This is a fallback parser that looks for function names in backticks.
    
    Args:
        content: Raw chat session content
        
    Returns:
        Parsed ChatSession object
    """
    session = ChatSession(raw_content=content, format=ChatFormat.GENERIC_MARKDOWN)
    
    # Find all backticked function names
    # Pattern: `function_name` or `namespace.function_name`
    tool_pattern = r'`([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)`'
    
    tool_matches = re.finditer(tool_pattern, content)
    seen_tools = set()
    tool_index = 1
    
    for match in tool_matches:
        full_name = match.group(1)
        if full_name not in seen_tools:
            seen_tools.add(full_name)
            tool_call = ToolCall(
                index=tool_index,
                full_name=full_name,
                parameters={},
                output=""
            )
            session.tool_calls.append(tool_call)
            tool_index += 1
    
    return session


def _detect_format(content: str) -> ChatFormat:
    """Auto-detect chat session format from content."""
    if "#### Tool Call:" in content or "#### Tool Response:" in content:
        return ChatFormat.MCP
    
    if "### User:" in content or "### Assistant:" in content:
        return ChatFormat.GENERIC_MARKDOWN
    
    return ChatFormat.UNKNOWN


def _parse_by_format(content: str, format: ChatFormat) -> ChatSession:
    """Parse content using the specified format."""
    if format == ChatFormat.MCP:
        return parse_mcp_format(content)
    
    if format == ChatFormat.GENERIC_MARKDOWN:
        return parse_generic_markdown(content)
    
    return ChatSession(raw_content=content, format=ChatFormat.UNKNOWN)


def parse_chat_session(content: str, format: Optional[ChatFormat] = None) -> ChatSession:
    """Parse chat session with automatic format detection.
    
    Args:
        content: Raw chat session content
        format: Optional format hint. If None, format is auto-detected.
        
    Returns:
        Parsed ChatSession object
    """
    # Check for custom parser first
    if format and format in _custom_parsers:
        return _custom_parsers[format](content)
    
    # Auto-detect format if not specified
    if format is None:
        format = _detect_format(content)
        if format == ChatFormat.UNKNOWN:
            return ChatSession(raw_content=content, format=ChatFormat.UNKNOWN)
    
    return _parse_by_format(content, format)

