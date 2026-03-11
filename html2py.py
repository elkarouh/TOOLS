#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML to Python converter using justhtml parser.

Converts HTML files to Python code that uses a context-manager-based XML builder.

Usage:
    python html2py.py                       # Show example page
    python html2py.py input.html            # Convert local HTML file (roundtrip)
    python html2py.py https://example.com   # Download and convert URL (roundtrip)
    python html2py.py URL -o generated.py   # Save generated Python code
    python html2py.py URL --no-preview      # Don't open browser preview
    python html2py.py URL --browser         # Use headless browser for SPAs
"""
from __future__ import annotations
from typing import Any
import argparse
import keyword
import ssl
import sys
import os
import io
import fileinput
import tempfile
import textwrap
import tokenize
import urllib.request
from xml.sax.saxutils import quoteattr as quote

from html.parser import HTMLParser
from dataclasses import dataclass, field

# Try to import justhtml as fallback, but we'll use our own parser
try:
    from justhtml import JustHTML
    HAS_JUSTHTML = True
except ImportError:
    HAS_JUSTHTML = False


@dataclass
class Node:
    """Simple HTML node representation."""
    name: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    data: str | None = None
    parent: 'Node | None' = None


class HTMLTreeBuilder(HTMLParser):
    """Build a tree of Node objects from HTML, preserving all elements."""

    def __init__(self):
        super().__init__()
        self.root = Node(name='#document')
        self.current = self.root

    def handle_decl(self, decl):
        """Handle DOCTYPE declaration."""
        node = Node(name='!doctype', data=decl, parent=self.current)
        self.current.children.append(node)

    def handle_starttag(self, tag, attrs):
        """Handle opening tag."""
        node = Node(name=tag, attrs=dict(attrs), parent=self.current)
        self.current.children.append(node)
        # Don't descend into void elements
        if tag.lower() not in VOID_ELEMENTS:
            self.current = node

    def handle_endtag(self, tag):
        """Handle closing tag."""
        # Walk up to find matching tag
        node = self.current
        while node and node.name != tag:
            node = node.parent
        if node and node.parent:
            self.current = node.parent

    def handle_startendtag(self, tag, attrs):
        """Handle self-closing tag like <br/>."""
        node = Node(name=tag, attrs=dict(attrs), parent=self.current)
        self.current.children.append(node)

    def handle_data(self, data):
        """Handle text content."""
        node = Node(name='#text', data=data, parent=self.current)
        self.current.children.append(node)

    def handle_comment(self, data):
        """Handle HTML comments."""
        node = Node(name='#comment', data=data, parent=self.current)
        self.current.children.append(node)


def parse_html(html_content: str) -> Node:
    """Parse HTML and return a tree of Node objects."""
    parser = HTMLTreeBuilder()
    parser.feed(html_content)
    return parser.root

def my_quote(content: str) -> str:
    """Quote content appropriately for Python string literals."""
    # Handle backslashes first to avoid double-escaping
    content = content.replace('\\', '\\\\')

    if "\n" in content:
        # For multiline content, use triple quotes and escape internal triple quotes
        if '"""' in content and "'''" in content:
            # Both triple quote styles present - escape one
            content = content.replace('"""', '\\"\\"\\"')
            return f'"""{content}"""'
        elif '"""' in content:
            return f"'''{content}'''"
        else:
            return f'"""{content}"""'

    # Single line content
    if '"' in content and "'" in content:
        # Both quote types - escape double quotes
        content = content.replace('"', '\\"')
        return f'"{content}"'
    if '"' in content:
        return f"'{content}'"
    return f'"{content}"'


#############################################################################################
############################# REDUCE NESTING LEVEL ##########################################
class G:
    """Global state for the converter."""
    id = 0
    funcs = {}
    indent_level = 0

remove_quotes = lambda stri: stri[1:-1]

def handle_token(tok):
    """Create a new function whenever the nesting level > MAX_NEST_LEVEL."""
    type, token, (srow, scol), _, _ = tok.type, tok.string, tok.start, tok.end, tok.line

    if tokenize.tok_name[type] in ('NL', 'NEWLINE'):
        if handle_token.state == 0:
            handle_token.out.append('\n')
        else:
            G.funcs[G.id].append('\n')
    elif tokenize.tok_name[type] == 'ENDMARKER':
        pass
    elif tokenize.tok_name[type] == 'INDENT':
        handle_token.current_indent_level += 1
        if handle_token.state == 0:
            if handle_token.current_indent_level > handle_token.MAX_NEST_LEVEL:
                handle_token.state = 1
                indent = '\t' * handle_token.current_indent_level
                G.id += 1
                func_name = 'FUNCTION_CALL%s' % G.id
                handle_token.out.append(indent + '%s(xml)\n' % func_name)
                G.funcs[G.id] = ["def %s(xml):\n" % func_name]
    elif tokenize.tok_name[type] == 'DEDENT':
        handle_token.current_indent_level -= 1
        if handle_token.state == 1:
            if handle_token.current_indent_level <= handle_token.MAX_NEST_LEVEL:
                handle_token.state = 0
    elif tokenize.tok_name[type] == 'COMMENT':
        tok = remove_quotes(repr(token))
        tok = tok.replace(r'\\', '\\')
        if tok.endswith(r'\n'):
            if srow > handle_token.current_line:
                if handle_token.state == 0:
                    indent = '\t' * handle_token.current_indent_level
                else:
                    indent = '\t' * (handle_token.current_indent_level - handle_token.MAX_NEST_LEVEL)
                handle_token.current_line = srow
            else:
                indent = ''
            handle_token.out.append(indent + '\t' * scol + tok[:-2] + '\n')
        else:
            handle_token.out.append(tok)
    else:
        tok = remove_quotes(repr(token))
        if tokenize.tok_name[type] == 'STRING':
            tok = tok.replace(r'\\', '\\')
            tok = tok[0] + tok[1:-1].replace(r"\'", "'") + tok[-1]
            if tok[:2] == r"\'" and tok[-2:] == r"\'":
                tok = tok[1] + tok[2:-2] + tok[-1]
            if tok.startswith('"""') or tok.startswith("'''"):
                tok = tok.replace(r'\n', '\n')
                tok = tok.replace(r'\t', '\t')
        if tok in keyword.kwlist:
            if tok in ('in', 'is', 'not', 'or', 'and'):
                tok = ' ' + tok + ' '
            elif tok in ('import',):
                if scol > 0:
                    tok = ' ' + tok + ' '
                else:
                    tok += ' '
            else:
                tok += ' '
        if srow > handle_token.current_line:
            if handle_token.state == 0:
                indent = '\t' * handle_token.current_indent_level
            else:
                indent = '\t' * (handle_token.current_indent_level - handle_token.MAX_NEST_LEVEL)
            handle_token.current_line = srow
        else:
            indent = ''
        if handle_token.state == 0:
            handle_token.out.append(indent + tok)
        else:
            G.funcs[G.id].append(indent + tok)

def reduce_nesting_level(infile: str, encoding: str, MAX_NEST_LEVEL: int = 15) -> str:
    """Reduce nesting level by extracting deeply nested code into functions."""
    G.funcs = {}  # Reset global state
    G.id = 0
    handle_token.out = []
    handle_token.state = 0
    handle_token.current_indent_level = 0
    handle_token.MAX_NEST_LEVEL = MAX_NEST_LEVEL
    handle_token.current_line = 0

    with open(infile, mode='rb') as f:
        tokens = tokenize.tokenize(f.readline)
        for tok in tokens:
            handle_token(tok)

    if G.funcs:  # a reduction took place
        output_file = infile + '3.py'
        with open(output_file, encoding=encoding, mode='w') as outfile:
            for id, fun in G.funcs.items():
                outfile.write("".join(fun))
            output = "".join(handle_token.out[1:])  # skip the utf-8 token
            outfile.write(output)
            print("REDUCED FILE created", output_file)
            return output_file
    return infile


def reduce_nesting_level_str(python_code: str, encoding: str = 'utf-8', MAX_NEST_LEVEL: int = 15) -> str:
    """Reduce nesting level of Python code string by extracting deeply nested code into functions."""
    import tempfile

    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', encoding=encoding, delete=False) as f:
        f.write(python_code)
        temp_file = f.name

    try:
        # Run reduction
        result_file = reduce_nesting_level(temp_file, encoding, MAX_NEST_LEVEL)

        # Read result
        with open(result_file, encoding=encoding) as f:
            result = f.read()

        # Clean up result file if different from input
        if result_file != temp_file:
            os.remove(result_file)

        return result
    finally:
        # Clean up temp file
        os.remove(temp_file)


#############################################################################################
# HTML to Python conversion using justhtml
#############################################################################################

# Self-closing tags in HTML5
VOID_ELEMENTS = frozenset([
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr'
])

INDENT = " "*4


def get_valid_tagname(tagname: str) -> str:
    """Convert tagname to valid Python attribute access syntax."""
    tagname = tagname + '_' if keyword.iskeyword(tagname) else tagname
    tagname = '.' + tagname if tagname.isidentifier() else '[' + quote(tagname) + ']'
    return tagname


def get_valid_attribute(attr: str) -> str:
    """Convert attribute name to valid Python identifier."""
    attr = attr + '_' if keyword.iskeyword(attr) else attr
    attr = attr.replace(':', '__', 1)
    if attr.isidentifier():
        return attr
    else:
        raise ValueError(f"Invalid attribute name: {attr}")


def get_attr_string(attrs: dict[str, Any] | None) -> str:
    """Convert attributes dict to Python function argument string."""
    if not attrs:
        return ""

    try:
        parts = []
        for key, val in attrs.items():
            valid_key = get_valid_attribute(key)
            if val is None or val == "":
                parts.append(f"{valid_key}=True")
            else:
                parts.append(f"{valid_key}={quote(val)}")
        return ", ".join(parts)
    except ValueError:
        # Fallback to **{} syntax for invalid identifiers
        parts = []
        for key, val in attrs.items():
            if val is None or val == "":
                parts.append(f"{quote(key)}:True")
            else:
                parts.append(f"{quote(key)}:{quote(val)}")
        return "**{" + ", ".join(parts) + "}"


def escape_non_printable(s: str) -> str:
    """Escape non-printable characters in a string."""
    output = ""
    for c in s:
        if c.isprintable() or c in "\n\t\r":
            output += c
        elif c <= '\xff':
            output += r'\x{0:02x}'.format(ord(c))
        else:
            output += c.encode('unicode_escape').decode('ascii')
    return output


def is_text_node(node: Any) -> bool:
    """Check if a node is a text node."""
    return getattr(node, 'name', '') == '#text'


def is_comment_node(node: Any) -> bool:
    """Check if a node is a comment node."""
    return getattr(node, 'name', '') == '#comment'


def is_doctype_node(node: Any) -> bool:
    """Check if a node is a doctype node."""
    name = getattr(node, 'name', '')
    return name.lower() == '!doctype' if name else False


def get_text_content(node: Any) -> str:
    """Get text content from a text node."""
    if hasattr(node, 'data') and node.data is not None:
        return node.data
    return ""


def node_to_python(node: Any, indent_level: int = 0, within_pre: bool = False) -> str:
    """Convert a justhtml node to Python code."""
    output = ""
    indent = INDENT * indent_level

    # Handle text nodes
    if is_text_node(node):
        text = get_text_content(node)
        if not within_pre:
            text = text.strip()
        if text:
            text = text.replace('&nbsp;', 'NBSP').replace('&#', 'ENT_NUM')
            if text.isprintable() or within_pre:
                quoted = my_quote(text)
            else:
                quoted = my_quote(escape_non_printable(text))
            output += f'{indent}xml.write({quoted})\n'
        return output

    # Handle comment nodes
    if is_comment_node(node):
        return output  # Skip comments for now

    # Handle doctype
    if is_doctype_node(node):
        return f'{indent}xml.write_doctype("html")\n'

    # Get node name safely
    node_name = getattr(node, 'name', '')
    node_attrs = getattr(node, 'attrs', {}) or {}
    node_children = getattr(node, 'children', []) or []

    # Skip document node, process children
    if node_name in ('#document', 'html'):
        if node_name == 'html':
            # Process html tag
            tagname = get_valid_tagname(node_name)
            attrs = get_attr_string(node_attrs) if node_attrs else ""

            if has_element_children(node):
                output += f'{indent}with xml{tagname}'
                output += f'({attrs}):\n' if attrs else ':\n'
                for child in node_children:
                    output += node_to_python(child, indent_level + 1, within_pre)
            else:
                # Childless node
                text = get_direct_text(node)
                if text:
                    quoted = my_quote(text)
                    output += f'{indent}xml{tagname}({quoted}'
                    output += f', {attrs})\n' if attrs else ')\n'
                else:
                    output += f'{indent}xml{tagname}({attrs})\n'
        else:
            # Document node - just process children
            for child in node_children:
                output += node_to_python(child, indent_level, within_pre)
        return output

    # Regular element node
    tagname = get_valid_tagname(node_name)
    attrs = get_attr_string(node_attrs) if node_attrs else ""
    is_pre = node_name == 'pre'
    new_within_pre = within_pre or is_pre

    # Check if it's a void/self-closing element
    if node_name.lower() in VOID_ELEMENTS:
        output += f'{indent}xml{tagname}(SELFCLOSING'
        output += f', {attrs})\n' if attrs else ')\n'
        return output

    # Check if node has element children
    if has_element_children(node):
        output += f'{indent}with xml{tagname}'
        output += f'({attrs}):\n' if attrs else ':\n'

        for child in node_children:
            output += node_to_python(child, indent_level + 1, new_within_pre)
    else:
        # Node with only text content or empty
        text = get_direct_text(node)
        if text:
            if not new_within_pre:
                text = text.strip()
            if text:
                text = text.replace('&nbsp;', 'NBSP').replace('&#', 'ENT_NUM')
                if text.isprintable() or new_within_pre:
                    quoted = my_quote(text)
                else:
                    quoted = my_quote(escape_non_printable(text))
                output += f'{indent}xml{tagname}({quoted}'
                output += f', {attrs})\n' if attrs else ')\n'
            else:
                output += f'{indent}xml{tagname}({attrs})\n'
        else:
            output += f'{indent}xml{tagname}({attrs})\n'

    return output


def has_element_children(node: Any) -> bool:
    """Check if node has any element (non-text) children."""
    children = getattr(node, 'children', []) or []
    for child in children:
        if not is_text_node(child) and not is_comment_node(child):
            return True
        # Also check if text node has substantial content
        if is_text_node(child):
            text = get_text_content(child).strip()
            # If there's text AND other element siblings, we have mixed content
            if text:
                for sibling in children:
                    if not is_text_node(sibling) and not is_comment_node(sibling):
                        return True
    return False


def get_direct_text(node: Any) -> str:
    """Get direct text content (from text node children only)."""
    texts = []
    children = getattr(node, 'children', []) or []
    for child in children:
        if is_text_node(child):
            texts.append(get_text_content(child))
    return "".join(texts)


def html_to_python(html_content: str, max_nest_level: int = 15, base_url: str = None) -> tuple[str, str]:
    """Convert HTML content to Python code.

    Args:
        html_content: The HTML string to convert
        max_nest_level: Maximum nesting level before extracting into functions (default 15)
        base_url: Optional base URL to inject into <head> for resolving relative URLs

    Returns:
        Tuple of (python_code, encoding)
    """
    # Use our custom parser that preserves all elements
    root = parse_html(html_content)

    # Reset global state
    G.indent_level = 0

    # Generate Python code from the parsed tree
    python_code = node_to_python(root)

    # Clean up the output
    python_code = python_code.replace('NBSP', '&nbsp;').replace('ENT_NUM', '&#')

    # Inject base tag into generated code to fix relative URLs (before reduction)
    if base_url:
        lines = python_code.split('\n')
        for i, line in enumerate(lines):
            if 'with xml.head' in line:
                indent = len(line) - len(line.lstrip()) + 4
                base_line = ' ' * indent + f'xml.base(SELFCLOSING, href="{base_url}")'
                lines.insert(i + 1, base_line)
                python_code = '\n'.join(lines)
                break

    # Reduce nesting level for complex documents
    # Wrap in a dummy structure so tokenizer can process it
    wrapped_code = f"with Builder() as xml:\n{textwrap.indent(python_code, '    ')}\n"
    reduced_code = reduce_nesting_level_str(wrapped_code, 'utf-8', max_nest_level)

    # Check if reduction occurred (functions were extracted)
    if reduced_code != wrapped_code:
        python_code = reduced_code

    return python_code, 'utf-8'


def to_python(data: str) -> tuple[str, str]:
    """Convert HTML data to a complete Python script."""
    python_code, encoding = html_to_python(data)
    return f"""\
from py2html import Builder, SELFCLOSING # , REMOTE, LOCAL
with Builder() as xml:
{textwrap.indent(python_code, ' ' * 4)}
output_filename = __file__ + ".html"
with open(output_filename, encoding="{encoding}", mode="w") as output_file:
    output_file.write(str(xml))

from http.server import HTTPServer, CGIHTTPRequestHandler
import os, webbrowser, threading

port = 8000

def start_server(path, port=8000):
    os.chdir(path)
    httpd = HTTPServer(('', port), CGIHTTPRequestHandler)
    httpd.serve_forever()

threading.Thread(
    name='daemon_server',
    target=start_server,
    args=('.', port),
    daemon=False
).start()

webbrowser.open(
    f"http://localhost:{{port}}/{{os.path.basename(output_filename)}}", new=2, autoraise=True)
""", encoding


DEBUG = False

# SSL context for URL downloads
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def download_html(url: str, use_browser: bool = False, wait_time: int = 3000) -> str:
    """Download HTML content from a URL.

    Args:
        url: The URL to download
        use_browser: If True, use Playwright headless browser to render JavaScript
        wait_time: Time in milliseconds to wait for JavaScript to execute (browser mode only)
    """
    if use_browser:
        return download_html_browser(url, wait_time)
    return download_html_simple(url)


def download_html_simple(url: str) -> str:
    """Download HTML content using urllib (no JavaScript execution)."""
    import gzip
    import zlib

    print(f"Downloading: {url}")
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; html2py/1.0)',
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'text/html,application/xhtml+xml'
        }
    )
    with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
        content = response.read()

        # Handle compressed content
        content_encoding = response.headers.get('Content-Encoding', '')
        if content_encoding == 'gzip':
            content = gzip.decompress(content)
        elif content_encoding == 'deflate':
            content = zlib.decompress(content)

        # Try to detect encoding from headers or default to utf-8
        encoding = response.headers.get_content_charset() or 'utf-8'
        return content.decode(encoding, errors='replace')


def download_html_browser(url: str, wait_time: int = 3000) -> str:
    """Download HTML content using Playwright headless browser (with JavaScript execution)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: Playwright not installed. Run: uv pip install playwright && uv run playwright install chromium", file=sys.stderr)
        sys.exit(1)
    from urllib.parse import urlparse

    print(f"Downloading with browser: {url}")
    print(f"Waiting {wait_time}ms for JavaScript to execute...")

    # Check for proxy settings from environment
    proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or \
                os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')

    with sync_playwright() as p:
        launch_options = {"headless": True}
        if proxy_url:
            parsed = urlparse(proxy_url)
            proxy_config = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
            if parsed.username and parsed.password:
                proxy_config["username"] = parsed.username
                proxy_config["password"] = parsed.password
            launch_options["proxy"] = proxy_config
            print(f"Using proxy: {parsed.hostname}:{parsed.port}")

        browser = p.chromium.launch(**launch_options)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(wait_time)
        html_content = page.content()
        browser.close()

    return html_content


def convert_and_run(html_content: str, base_url: str = "", output_file: str = None) -> str:
    """Convert HTML to Python, execute it, and return regenerated HTML."""
    from py2html import Builder, SELFCLOSING, NOCONTENT

    python_code, encoding = html_to_python(html_content, base_url=base_url)

    print("\n" + "=" * 60)
    print("Generated Python code:")
    print("=" * 60)
    lines = python_code.split('\n')
    for i, line in enumerate(lines[:50]):
        print(f"  {line}")
    if len(lines) > 50:
        print(f"  ... ({len(lines) - 50} more lines)")
    print("=" * 60 + "\n")

    # Check if reduction occurred (code already has wrapper and extracted functions)
    has_wrapper = 'with Builder()' in python_code

    # Save generated Python code to file if specified
    if output_file:
        output_path = os.path.abspath(output_file)

        if has_wrapper:
            full_script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated HTML builder script.
Run this file to generate HTML and open it in the browser.
"""
from py2html import Builder, SELFCLOSING, NOCONTENT, preview_in_browser

{python_code}

html_output = str(xml)
preview_in_browser(html_output, "generated_output.html")
print("HTML opened in browser!")
'''
        else:
            indented_code = '\n'.join('    ' + line for line in python_code.split('\n'))
            full_script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated HTML builder script.
Run this file to generate HTML and open it in the browser.
"""
from py2html import Builder, SELFCLOSING, NOCONTENT, preview_in_browser

with Builder() as xml:
{indented_code}

html_output = str(xml)
preview_in_browser(html_output, "generated_output.html")
print("HTML opened in browser!")
'''
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_script)
        print(f"Saved generated Python code to: {output_path}\n")

    # Build the executable code
    if has_wrapper:
        exec_code = f'''
from py2html import Builder, SELFCLOSING, NOCONTENT

{python_code}

result = str(xml)
'''
    else:
        indented_code = '\n'.join('    ' + line for line in python_code.split('\n'))
        exec_code = f'''
from py2html import Builder, SELFCLOSING, NOCONTENT

with Builder() as xml:
{indented_code}

result = str(xml)
'''

    # Execute the generated code
    namespace = {}
    exec(exec_code, namespace)

    return namespace['result']


def is_url(s: str) -> bool:
    """Check if a string looks like a URL."""
    return s.startswith('http://') or s.startswith('https://')


def run_example() -> None:
    """Run the original example page behavior (no arguments given)."""
    infile = "test.html"
    data = html_text

    infile_directory = os.path.dirname(infile)
    print(f"{infile_directory=} {infile=}")

    output, ENCODING = to_python(data)
    output_file = infile + '.py'

    if DEBUG:
        print(output)
    else:
        with io.open(output_file, mode="w") as outfile:
            outfile.write(output)

        output_file = reduce_nesting_level(output_file, ENCODING)

        with fileinput.input(output_file, inplace=True) as f:
            for line in f:
                if f.isfirstline():
                    print('#!/usr/bin/env python3')
                    print(f"# -*- coding: {ENCODING} -*-")
                print(line, end="")

        print('HTML FILE CREATED:', output_file)

        def make_executable(path):
            mode = os.stat(path).st_mode
            mode |= (mode & 0o444) >> 2  # copy R bits to X
            os.chmod(path, mode)

        make_executable(output_file)
        os.system(output_file)


def run_roundtrip(args) -> None:
    """Run the roundtrip: download/read HTML -> convert to Python -> execute -> preview."""
    from py2html import preview_in_browser

    try:
        source = args.source

        # Determine if source is a URL or a file
        if is_url(source):
            original_html = download_html(source, use_browser=args.browser, wait_time=args.wait)
            base_url = source
        else:
            # It's a file path
            print(f"Reading: {source}")
            try:
                with open(source, encoding='utf-8') as f:
                    original_html = f.read()
            except FileNotFoundError:
                print(f"Error: File '{source}' not found", file=sys.stderr)
                sys.exit(1)
            except (UnicodeDecodeError, IOError) as e:
                print(f"Error reading '{source}': {e}", file=sys.stderr)
                sys.exit(1)
            base_url = ""

        print(f"Read {len(original_html)} bytes")

        # Convert to Python and regenerate HTML
        print("\nConverting HTML to Python and regenerating...")
        regenerated_html = convert_and_run(original_html, base_url=base_url, output_file=args.output)
        print(f"Regenerated {len(regenerated_html)} bytes of HTML")

        # Preview in browser (unless --no-preview)
        if not args.no_preview:
            print("\nOpening regenerated HTML in browser...")
            filepath = preview_in_browser(regenerated_html, "roundtrip_result.html")
            print(f"Preview file: {filepath}")

        # Save original for comparison
        original_file = tempfile.gettempdir() + "/roundtrip_original.html"
        with open(original_file, 'w', encoding='utf-8') as f:
            f.write(original_html)
        print(f"Original saved to: {original_file}")

        print("\nRound-trip complete!")

    except urllib.error.URLError as e:
        print(f"Error downloading URL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point - no args shows example page, with args performs roundtrip."""
    parser = argparse.ArgumentParser(
        description="HTML to Python converter. No arguments shows an example page. "
                    "With a URL or HTML file, performs a roundtrip conversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                                    # Show example page
    %(prog)s input.html                         # Roundtrip a local HTML file
    %(prog)s https://example.com                # Roundtrip a URL
    %(prog)s https://example.com -o generated.py
    %(prog)s https://spa-site.com --browser     # Use headless browser for SPAs
"""
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="URL or HTML file to convert (omit to show example page)"
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Save generated Python code to specified file"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Don't open browser preview"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Use headless browser (Playwright) to render JavaScript before capturing HTML"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=3000,
        metavar="MS",
        help="Time in milliseconds to wait for JavaScript (browser mode only, default: 3000)"
    )
    args = parser.parse_args()

    if args.source is None:
        # No argument given: show example page
        run_example()
    else:
        # URL or file given: perform roundtrip
        run_roundtrip(args)


#########################################################################
# Test HTML samples
html_text = """\
<html lang="en">
<head>
    <title>Test Page</title>
</head>
<body class="antialiased bg-gray-100">
    <h1>Hello World</h1>
    <p>This is a <strong>test</strong> paragraph.</p>
    <br/>
    <img src="test.png" alt="Test image"/>
    <div id="main">
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    main()
