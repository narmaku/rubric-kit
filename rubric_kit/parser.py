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
    
    # Pattern to match tool calls
    # Matches: #### Tool Call: `function_name` (namespace: `namespace_name`)
    # or: #### Tool Call: `function_name`
    tool_call_pattern = r'#### Tool Call:\s*`([^`]+)`(?:\s*\(namespace:\s*`([^`]+)`\))?'
    
    # Pattern to match tool responses
    tool_response_pattern = r'#### Tool Response:\s*\n(.*?)(?=\n####|###|$)'
    
    # Pattern to match assistant responses
    assistant_pattern = r'### Assistant:\s*\n(.*?)(?=\n###|$)'
    
    # Find all tool calls
    tool_call_matches = list(re.finditer(tool_call_pattern, content, re.MULTILINE | re.DOTALL))
    
    tool_index = 1
    for match in tool_call_matches:
        function_name = match.group(1).strip()
        namespace = match.group(2).strip() if match.group(2) else None
        
        # Build full name
        if namespace:
            full_name = f"{namespace}.{function_name}"
        else:
            full_name = function_name
        
        # Find arguments section after this tool call
        args_start = match.end()
        args_end = content.find("#### Tool Response:", args_start)
        if args_end == -1:
            args_end = content.find("---", args_start)
        if args_end == -1:
            args_end = len(content)
        
        args_section = content[args_start:args_end]
        
        # Parse arguments
        parameters = {}
        if "**Arguments:**" in args_section:
            args_text = args_section.split("**Arguments:**", 1)[1]
            
            # Check for empty object
            if "*empty object*" in args_text:
                parameters = {}
            else:
                # Parse argument list items
                # Format: *   **arg_name**: value
                arg_pattern = r'\*\s+\*\*([^:]+)\*\*:\s*(.*?)(?=\n\*|$)'
                arg_matches = re.finditer(arg_pattern, args_text, re.MULTILINE | re.DOTALL)
                
                for arg_match in arg_matches:
                    arg_name = arg_match.group(1).strip()
                    arg_value = arg_match.group(2).strip()
                    
                    # Handle null values
                    if arg_value == "_null_" or arg_value == "":
                        arg_value = None
                    else:
                        # Remove leading/trailing whitespace and newlines
                        arg_value = arg_value.strip()
                    
                    parameters[arg_name] = arg_value
        
        # Find tool response
        response_start = content.find("#### Tool Response:", match.start())
        output = ""
        if response_start != -1:
            response_start += len("#### Tool Response:")
            # Find next section marker or end
            response_end = content.find("\n####", response_start)
            if response_end == -1:
                response_end = content.find("\n###", response_start)
            if response_end == -1:
                response_end = len(content)
            
            output = content[response_start:response_end].strip()
            # Remove leading/trailing dashes
            output = re.sub(r'^---+', '', output)
            output = re.sub(r'---+$', '', output)
            output = output.strip()
        
        tool_call = ToolCall(
            index=tool_index,
            full_name=full_name,
            parameters=parameters,
            output=output
        )
        session.tool_calls.append(tool_call)
        tool_index += 1
    
    # Find assistant responses
    assistant_matches = list(re.finditer(assistant_pattern, content, re.MULTILINE | re.DOTALL))
    response_index = 1
    for match in assistant_matches:
        response_content = match.group(1).strip()
        # Remove tool call sections from assistant response
        # (they're already captured separately)
        response_content = re.sub(r'#### Tool Call:.*?#### Tool Response:.*?(?=\n####|\n###|$)', '', 
                                 response_content, flags=re.DOTALL)
        response_content = re.sub(r'#### Tool Call:.*?(?=\n####|\n###|$)', '', 
                                 response_content, flags=re.DOTALL)
        response_content = response_content.strip()
        
        if response_content:
            # Remove redacted reasoning blocks
            response_content = re.sub(r'<think>.*?</think>', '', 
                                    response_content, flags=re.DOTALL)
            response_content = response_content.strip()
            
            if response_content:
                session.assistant_responses.append(
                    AssistantResponse(index=response_index, content=response_content)
                )
                response_index += 1
    
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
        # Check for MCP format markers
        if "#### Tool Call:" in content or "#### Tool Response:" in content:
            format = ChatFormat.MCP
        # Check if it's actually a chat session (has User/Assistant markers)
        elif "### User:" in content or "### Assistant:" in content:
            # It's a chat session but not MCP format - try generic markdown
            format = ChatFormat.GENERIC_MARKDOWN
        else:
            # Unknown format - likely not a chat session, return empty session
            return ChatSession(raw_content=content, format=ChatFormat.UNKNOWN)
    
    # Parse based on detected/specified format
    if format == ChatFormat.MCP:
        return parse_mcp_format(content)
    elif format == ChatFormat.GENERIC_MARKDOWN:
        return parse_generic_markdown(content)
    else:
        # Unknown format - return empty session
        return ChatSession(raw_content=content, format=ChatFormat.UNKNOWN)

