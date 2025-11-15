"""Tests for chat session parser."""

import pytest
from rubric_kit.parser import (
    ChatSession,
    ToolCall,
    AssistantResponse,
    ChatFormat,
    parse_chat_session,
    parse_mcp_format,
    parse_generic_markdown,
    register_custom_parser,
)


def test_tool_call_post_init():
    """Test that ToolCall automatically extracts namespace and function."""
    tc = ToolCall(
        index=1,
        full_name="linux_diagnostics.get_system_information"
    )
    assert tc.namespace == "linux_diagnostics"
    assert tc.function == "get_system_information"
    
    tc_no_namespace = ToolCall(
        index=1,
        full_name="simple_function"
    )
    assert tc_no_namespace.namespace is None
    assert tc_no_namespace.function == "simple_function"


def test_chat_session_get_tool_call_sequence():
    """Test getting ordered list of tool names."""
    session = ChatSession(raw_content="test")
    session.tool_calls = [
        ToolCall(index=2, full_name="tool_b"),
        ToolCall(index=1, full_name="tool_a"),
        ToolCall(index=3, full_name="tool_c"),
    ]
    
    sequence = session.get_tool_call_sequence()
    assert sequence == ["tool_a", "tool_b", "tool_c"]


def test_chat_session_get_tool_by_name():
    """Test finding tools by name."""
    session = ChatSession(raw_content="test")
    session.tool_calls = [
        ToolCall(index=1, full_name="linux.get_cpu"),
        ToolCall(index=2, full_name="linux.get_memory"),
        ToolCall(index=3, full_name="network.get_interfaces"),
    ]
    
    linux_tools = session.get_tool_by_name("linux")
    assert len(linux_tools) == 2
    assert all("linux" in tc.full_name for tc in linux_tools)


def test_chat_session_get_final_response():
    """Test getting the last assistant response."""
    session = ChatSession(raw_content="test")
    session.assistant_responses = [
        AssistantResponse(index=1, content="First response"),
        AssistantResponse(index=2, content="Second response"),
    ]
    
    assert session.get_final_response() == "Second response"
    
    empty_session = ChatSession(raw_content="test")
    assert empty_session.get_final_response() is None


def test_parse_mcp_format():
    """Test parsing MCP-style chat sessions."""
    mcp_content = """
### Assistant:
#### Tool Call: `get_system_information` (namespace: `linux_diagnostics`)
**Arguments:**
*   **host**: 
    _null_
*   **username**: 
    _null_

---

#### Tool Response:
Operating System: Fedora Linux 42
Kernel: 6.16.7-200.fc42.x86_64

---

### Assistant:
#### Tool Call: `get_cpu_information` (namespace: `linux_diagnostics`)
**Arguments:**
*   **host**: 
    _null_

---

#### Tool Response:
CPU Model: AMD Ryzen 7 PRO 7840HS
Cores: 8

---

### Assistant:
The system is running Fedora Linux 42 with an AMD Ryzen 7 PRO 7840HS processor.
"""
    
    session = parse_mcp_format(mcp_content)
    
    assert session.format == ChatFormat.MCP
    assert len(session.tool_calls) == 2
    
    # Check first tool call
    assert session.tool_calls[0].full_name == "linux_diagnostics.get_system_information"
    assert session.tool_calls[0].namespace == "linux_diagnostics"
    assert session.tool_calls[0].function == "get_system_information"
    assert session.tool_calls[0].parameters == {"host": None, "username": None}
    assert "Fedora Linux 42" in session.tool_calls[0].output
    
    # Check second tool call
    assert session.tool_calls[1].full_name == "linux_diagnostics.get_cpu_information"
    assert session.tool_calls[1].parameters == {"host": None}
    assert "AMD Ryzen 7" in session.tool_calls[1].output
    
    # Check assistant response
    assert len(session.assistant_responses) > 0
    final_response = session.get_final_response()
    assert "Fedora Linux 42" in final_response
    assert "AMD Ryzen 7" in final_response


def test_parse_generic_markdown():
    """Test parsing generic markdown format."""
    generic_content = """
The assistant called `fetch_weather_data` with location "San Francisco".

Then it invoked `get_temperature` to retrieve the current temperature.

Finally, it responded: The weather in San Francisco is 72°F and sunny.
"""
    
    session = parse_generic_markdown(generic_content)
    
    assert session.format == ChatFormat.GENERIC_MARKDOWN
    assert len(session.tool_calls) >= 2
    
    tool_names = [tc.full_name for tc in session.tool_calls]
    assert "fetch_weather_data" in tool_names
    assert "get_temperature" in tool_names


def test_parse_chat_session_auto_detect():
    """Test automatic format detection."""
    mcp_content = """
#### Tool Call: `some_function` (namespace: `some_namespace`)

#### Tool Response:
Some output
"""
    
    session = parse_chat_session(mcp_content)
    assert session.format == ChatFormat.MCP


def test_parse_chat_session_explicit_format():
    """Test parsing with explicit format hint."""
    content = "Some content"
    
    session = parse_chat_session(content, format=ChatFormat.GENERIC_MARKDOWN)
    assert session.format == ChatFormat.GENERIC_MARKDOWN


def test_mcp_format_no_namespace():
    """Test MCP format without namespace."""
    content = """
#### Tool Call: `simple_function`
**Arguments:**
*   **arg1**: 
    value1

---

#### Tool Response:
Result data
"""
    
    session = parse_mcp_format(content)
    assert len(session.tool_calls) == 1
    assert session.tool_calls[0].full_name == "simple_function"
    assert session.tool_calls[0].namespace is None
    assert session.tool_calls[0].function == "simple_function"
    assert session.tool_calls[0].parameters["arg1"] == "value1"


def test_empty_chat_session():
    """Test parsing empty or minimal content."""
    session = parse_chat_session("Some plain text with no tool calls")
    assert len(session.tool_calls) == 0


def test_register_custom_parser():
    """Test registering a custom parser."""
    def custom_parser(content: str) -> ChatSession:
        session = ChatSession(raw_content=content, format=ChatFormat.UNKNOWN)
        session.tool_calls.append(ToolCall(index=1, full_name="custom_tool"))
        return session
    
    custom_format = ChatFormat.UNKNOWN
    register_custom_parser(custom_format, custom_parser)
    
    # Note: This test just verifies registration doesn't crash
    # Actual usage would require adding to the format enum


def test_tool_call_sequence_preservation():
    """Test that tool call order is preserved."""
    mcp_content = """
#### Tool Call: `first_tool`
---
#### Tool Response:
First output
---

#### Tool Call: `second_tool`
---
#### Tool Response:
Second output
---

#### Tool Call: `third_tool`
---
#### Tool Response:
Third output
"""
    
    session = parse_mcp_format(mcp_content)
    sequence = session.get_tool_call_sequence()
    
    assert sequence == ["first_tool", "second_tool", "third_tool"]
    assert session.tool_calls[0].index == 1
    assert session.tool_calls[1].index == 2
    assert session.tool_calls[2].index == 3

