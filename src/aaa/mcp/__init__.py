"""MCP (Model Context Protocol) client and server: use any MCP server's tools, and MCP memory servers, in AAA."""

from aaa.mcp import wire
from aaa.mcp.builder import MCPServerBuilder
from aaa.mcp.client import MCPClient, MCPToolInfo, mcp_tool_name, render_content
from aaa.mcp.compile import acompile_spec, compile_spec
from aaa.mcp.config import load_spec
from aaa.mcp.memory import mcp_memory
from aaa.mcp.reload import ReloadableServer
from aaa.mcp.server import MCPToolServer, serve_stdio

__all__ = [
    "MCPClient",
    "MCPServerBuilder",
    "MCPToolInfo",
    "MCPToolServer",
    "ReloadableServer",
    "acompile_spec",
    "compile_spec",
    "load_spec",
    "mcp_memory",
    "mcp_tool_name",
    "render_content",
    "serve_stdio",
    "wire",
]
