# -*- coding: utf-8 -*-
"""
py2html - A Python library for generating HTML using context managers.

Usage:
    from py2html import Builder, SELFCLOSING

    with Builder() as xml:
        xml.write_doctype("html")
        with xml.html(lang="en"):
            with xml.head:
                xml.title("My Page")
            with xml.body:
                xml.h1("Hello World")
                xml.p("This is a paragraph.")
                xml.br(SELFCLOSING)
                xml.img(SELFCLOSING, src="image.png", alt="An image")

    print(str(xml))
"""
from __future__ import annotations
from typing import Any, Callable
import os
import re
import sys
import tempfile
import webbrowser
from io import StringIO
from xml.sax import saxutils
import html
import functools
import inspect
import textwrap
from keyword import kwlist as PYTHON_KWORD_LIST

__all__ = ['Builder', 'Namespace', 'SELFCLOSING', 'NOCONTENT', 'py2html',
           'preview_html', 'preview_in_browser', 'LOCAL', 'REMOTE']

########################################################################
# Constants
########################################################################
SELFCLOSING = object()
NOCONTENT = object()
LOCAL = object()
REMOTE = object()

# HTML5 void elements (self-closing tags)
SELFCLOSINGTAGS = frozenset([
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr'
])


########################################################################
# Utility functions
########################################################################
def escape(txt: str) -> str:
    """Escape HTML special characters while preserving &nbsp; and numeric entities."""
    if not isinstance(txt, str):
        txt = str(txt)
    # Preserve special sequences
    txt2 = txt.replace('&nbsp;', "XXXXX").replace('&#', 'YYYYY')
    text3 = html.escape(txt2)
    output = text3.replace("XXXXX", '&nbsp;').replace('YYYYY', '&#')
    return output


class Stack(list):
    """Simple stack implementation."""
    push = list.append

    def peek(self) -> Any | None:
        return self[-1] if self else None


########################################################################
# Optional pscript integration for Python-to-JavaScript translation
########################################################################
try:
    from pscript import py2js
    import ast

    def extract_python_variable_names(code):
        """Extract variable names starting with 'py_' from Python code."""
        tree = ast.parse(code)
        variable_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith("py_"):
                variable_names.add(node.id[3:])  # remove the prefix
        return variable_names

    def translate2javascript(func):
        """Decorator to translate Python function body to JavaScript."""
        def get_python_function_body(ob):
            lines, linenr = inspect.getsourcelines(ob)
            # Normalize indentation
            indent = len(lines[0]) - len(lines[0].lstrip())
            for i in range(1, len(lines)):
                line = lines[i].rstrip()
                line_indent = len(line) - len(line.lstrip())
                if line_indent < indent and line.strip():
                    assert line.lstrip().startswith('#')
                    lines[i] = indent * ' ' + line.lstrip()
                else:
                    lines[i] = line[indent:]
            # Skip decorators
            while not lines[0].lstrip().startswith('def'):
                lines.pop(0)
            lines.pop(0)  # skip the 'def' line
            pycode = '\n'.join(lines)
            return textwrap.dedent(pycode)

        def wrapper(*args, **kwargs):
            function_body = get_python_function_body(func)
            pyvars = extract_python_variable_names(function_body)
            for pyvar in pyvars:
                if pyvar in kwargs:
                    function_body = function_body.replace('py_' + pyvar, str(kwargs[pyvar]))
            return py2js(function_body)
        return wrapper

    PSCRIPT_AVAILABLE = True
except ImportError:
    PSCRIPT_AVAILABLE = False
    translate2javascript = None


########################################################################
# Decorators
########################################################################
def py2html(func):
    """
    Decorator to have a function return HTML.
    The xml argument is a placeholder, will be replaced by own Builder.
    Don't include the first parameter when calling!

    Example:
        @py2html
        def my_component(xml, title, items):
            with xml.div(class_="component"):
                xml.h2(title)
                with xml.ul:
                    for item in items:
                        xml.li(item)

        html = my_component("My List", ["Item 1", "Item 2"])
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Builder() as xml:
            func(xml, *args, **kwargs)
        return str(xml)
    return wrapper


########################################################################
# Preview functions
########################################################################
def preview_in_browser(html_content: str, filename: str | None = None) -> str:
    """
    Preview HTML content in the default web browser.

    Args:
        html_content: HTML string to preview
        filename: Optional filename for the temp file (default: auto-generated)

    Returns:
        Path to the created temporary file
    """
    if filename is None:
        fd, filepath = tempfile.mkstemp(suffix='.html', prefix='py2html_preview_')
        os.close(fd)
    else:
        filepath = os.path.join(tempfile.gettempdir(), filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    webbrowser.open(f'file://{filepath}')
    return filepath


def preview_html(html_content: str, method: str = 'browser', title: str = "Preview") -> str | tuple[str, str] | None:
    """
    Preview HTML content using the specified method.

    Args:
        html_content: HTML string to preview
        method: 'browser' (default), 'webview', or 'server'
        title: Window title (for webview method)

    Returns:
        For 'browser': the path to the temp file
        For 'webview': None (blocks until window is closed)
        For 'server': tuple of (filepath, server_url)
    """
    if method == 'browser':
        return preview_in_browser(html_content)

    elif method == 'webview':
        try:
            import webview
            window = webview.create_window(title, html=html_content)
            webview.start(debug=True)
        except ImportError:
            print("pywebview not installed. Install with: pip install pywebview")
            print("Falling back to browser preview...")
            return preview_in_browser(html_content)
        except AttributeError as e:
            # Handle case where webview module exists but isn't pywebview
            print(f"webview error: {e}")
            print("Make sure pywebview is installed: pip install pywebview")
            print("Falling back to browser preview...")
            return preview_in_browser(html_content)
        except Exception as e:
            # Handle GTK/WebKit library errors (common on Linux)
            error_msg = str(e)
            print(f"webview failed: {error_msg}")
            if 'webkit' in error_msg.lower() or 'gtk' in error_msg.lower():
                print("\nThis appears to be a WebKit/GTK library issue on your system.")
                print("Common fixes:")
                print("  - Install WebKit: sudo apt install libwebkit2gtk-4.0-dev  # Debian/Ubuntu")
                print("  - Install WebKit: sudo dnf install webkit2gtk3-devel     # Fedora/RHEL")
                print("  - Or use browser preview instead (recommended)")
            print("\nFalling back to browser preview...")
            return preview_in_browser(html_content)

    elif method == 'server':
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import threading

        # Save to temp file
        fd, filepath = tempfile.mkstemp(suffix='.html', prefix='py2html_preview_')
        os.close(fd)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Start server
        port = 8000
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

        server = HTTPServer(('localhost', port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        url = f'http://localhost:{port}/{filename}'
        webbrowser.open(url)
        return filepath, url

    else:
        raise ValueError(f"Unknown preview method: {method}")


########################################################################
# Main Builder class
########################################################################
class Builder:
    """
    HTML Builder using context managers for nested elements.

    Example:
        with Builder() as xml:
            with xml.div(class_="container"):
                xml.h1("Title")
                xml.p("Paragraph text")

        print(str(xml))  # <div class="container"><h1>Title</h1><p>Paragraph text</p></div>
    """

    def __init__(self, indent_char: str = ' ' * 4, compact: bool = False) -> None:
        """
        Initialize the Builder.

        Args:
            indent_char: String to use for indentation (default: 4 spaces)
            compact: If True, no indentation or newlines (default: False)
        """
        self._document = StringIO()
        self._indentation = 0
        self._indent = '' if compact else indent_char
        self.namespace = None
        self.stack = Stack()
        self.root = None
        self.within_pre_tag = False
        self.ENDLINE_CHAR = '' if compact else '\n'

    def __getattr__(self, name: str) -> Element:
        """Create a new element with the given tag name."""
        new_elem = Element(name, self)
        if self.root is None:
            self.root = new_elem
        return new_elem

    __getitem__ = __getattr__

    def __str__(self) -> str:
        """Return the generated HTML."""
        return self._document.getvalue()

    def __enter__(self) -> Builder:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def _write_indent_nl(self, line: str) -> None:
        """Write a line with indentation and newline."""
        if self.within_pre_tag:
            self._document.write(line)
        else:
            self._document.write(f"{self._indentation * self._indent}{line}{self.ENDLINE_CHAR}")

    def write(self, value: str) -> None:
        """Write escaped text with indentation."""
        self._write_indent_nl(escape(value))

    write_indent_nl_esc = write  # Alias for compatibility

    def write_raw(self, value: str) -> None:
        """Write text without indentation (escaped)."""
        self.write_escaped(value)

    def write_indent_nl(self, value: str) -> None:
        """Write text with indentation (not escaped)."""
        self._write_indent_nl(value)

    def write_no_escape(self, value: str) -> None:
        """Write text without escaping or indentation."""
        self._document.write(value)

    def write_escaped(self, line: str) -> None:
        """Write escaped text without indentation."""
        if line.startswith('<') and line.endswith('>'):
            line = '<' + escape(line[1:-1]) + '>'
        else:
            line = escape(line)
        if line.strip():
            self._document.write(line)

    def write_escaped_nl(self, line: str) -> None:
        """Write escaped text with newline."""
        self.write_escaped(line)
        self._document.write(self.ENDLINE_CHAR)

    def write_doctype(self, doctype: str = "html") -> None:
        """Write DOCTYPE declaration."""
        self._document.write(f"<!DOCTYPE {doctype}>{self.ENDLINE_CHAR}")

    def comment(self, text: str) -> None:
        """Write an HTML comment."""
        self._write_indent_nl(f"<!-- {text} -->")


########################################################################
# Namespace support
########################################################################
class Namespace:
    """Context manager for XML namespaces."""

    def __init__(self, builder: Builder, tagname: str, namespace: str) -> None:
        self.builder = builder
        self.tagname = tagname
        self.namespace = namespace

    def __enter__(self) -> Namespace:
        self.builder.namespace = (self.tagname, self.namespace)
        return self

    def __exit__(self, *args: Any) -> None:
        self.builder.namespace = None


########################################################################
# Element class
########################################################################
class Element:
    """Represents an HTML element."""

    PYTHON_KWORD_MAP: dict[str, str] = {k + '_': k for k in PYTHON_KWORD_LIST}

    def __init__(self, name: str, builder: Builder) -> None:
        self.builder = builder
        self.tagname = self.nameprep(name)
        self.attrs: dict[str, Any] = {}
        self.no_text_and_maybechildren = False
        self.text: Any = None
        self.closingtag: str | None = None

    def __enter__(self) -> Element:
        """Enter context - write opening tag."""
        if self.tagname == 'pre':
            self.builder.within_pre_tag = True

        if not self.no_text_and_maybechildren:
            attrs = self.serialized_attrs
            if attrs:
                self.builder._write_indent_nl(f'<{self.tagname} {attrs}>')
            else:
                self.builder._write_indent_nl(f'<{self.tagname}>')
        else:
            # Remove the closing tag we already wrote
            # The buffer contains: <tag attrs></tag> or <tag attrs></tag>\n
            # We want to truncate to: <tag attrs>
            buffer_content = self.builder._document.getvalue()
            # Find and remove the closing tag (and any trailing newline)
            closing_tag_with_newline = self.closingtag + self.builder.ENDLINE_CHAR
            if buffer_content.endswith(closing_tag_with_newline):
                new_len = len(buffer_content) - len(closing_tag_with_newline)
            elif buffer_content.endswith(self.closingtag):
                new_len = len(buffer_content) - len(self.closingtag)
            else:
                new_len = len(buffer_content)
            self.builder._document.seek(new_len)
            self.builder._document.truncate()
            self.builder._document.write(self.builder.ENDLINE_CHAR)

        self.builder._indentation += 1

        if self.builder.stack:
            parent = self.builder.stack.peek()
        else:
            self.builder.root = self

        self.builder.stack.push(self)
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        """Exit context - write closing tag."""
        if self.tagname == 'pre':
            self.builder.within_pre_tag = False

        self.builder._indentation -= 1
        self.builder._write_indent_nl('</%s>' % self.tagname)
        self.builder.stack.pop()

    def __call__(self, text_content: Any = NOCONTENT, *bool_attrs: Any, **attrs: Any) -> Element:
        """
        Create the element with optional content and attributes.

        Args:
            text_content: Text content or SELFCLOSING/NOCONTENT
            *bool_attrs: Boolean attributes (e.g., 'defer', 'async')
            **attrs: Key-value attributes

        Returns:
            self (for use with 'with' statement if needed)
        """
        self.attrs.update(attrs)
        attributes = self.serialized_attrs

        if bool_attrs:
            bool_attr_str = ' '.join(str(a) for a in bool_attrs if a is not SELFCLOSING)
            if bool_attr_str:
                attributes += ' ' + bool_attr_str

        # Handle self-closing tags
        is_void = self.tagname.lower() in SELFCLOSINGTAGS
        is_selfclosing = text_content is SELFCLOSING or (text_content is NOCONTENT and is_void)

        if is_selfclosing:
            if self.tagname.lower() in ('script', 'style'):
                # Script and style need closing tags even when empty
                content = '' if text_content is SELFCLOSING else ''
                if attributes:
                    self.builder._write_indent_nl(f'<{self.tagname} {attributes}>{content}</{self.tagname}>')
                else:
                    self.builder._write_indent_nl(f'<{self.tagname}>{content}</{self.tagname}>')
            elif self.tagname.lower() == 'meta':
                # Meta tags don't need closing slash in HTML5
                if attributes:
                    self.builder._write_indent_nl(f'<meta {attributes}>')
                else:
                    self.builder._write_indent_nl('<meta>')
            else:
                # Standard self-closing tag
                if attributes:
                    self.builder._write_indent_nl(f'<{self.tagname} {attributes}/>')
                else:
                    self.builder._write_indent_nl(f'<{self.tagname}/>')
            self.text = ""

        elif text_content is not NOCONTENT:
            # Element with text content
            if self.tagname.lower() in ('script', 'style'):
                # Don't escape script/style content
                if attributes:
                    self.builder.write_indent_nl(f'<{self.tagname} {attributes}>{text_content}</{self.tagname}>')
                else:
                    self.builder.write_indent_nl(f'<{self.tagname}>{text_content}</{self.tagname}>')
            else:
                escaped_content = escape(str(text_content))
                if attributes:
                    self.builder.write_indent_nl(f'<{self.tagname} {attributes}>{escaped_content}</{self.tagname}>')
                else:
                    self.builder.write_indent_nl(f'<{self.tagname}>{escaped_content}</{self.tagname}>')
            self.text = text_content

        else:
            # Empty element (may have children added via 'with')
            if attributes:
                self.builder.write_indent_nl(f'<{self.tagname} {attributes}></{self.tagname}>')
            else:
                self.builder.write_indent_nl(f'<{self.tagname}></{self.tagname}>')
            self.text = text_content
            self.closingtag = f'</{self.tagname}>'
            self.no_text_and_maybechildren = True

        return self

    @property
    def serialized_attrs(self) -> str:
        """Serialize attributes to HTML string."""
        ns = []
        if isinstance(self.builder.namespace, tuple):
            if self.builder.namespace[1] is not None:
                ns = [f'xmlns:{self.builder.namespace[0]}="{self.builder.namespace[1]}"']
                self.builder.namespace = (self.builder.namespace[0], None)

        def att_quote(val):
            val = str(val).replace('&#', 'QQQQQ')
            return saxutils.quoteattr(val).replace('QQQQQ', '&#')

        serialized = []
        for attr, value in self.attrs.items():
            attr_name = self.nameprep(attr)
            if value is True:
                serialized.append(f'{attr_name}="true"')
            elif value is False:
                serialized.append(f'{attr_name}=false')
            else:
                serialized.append(f'{attr_name}={att_quote(value)}')

        result = ns + serialized
        return ' '.join(result)

    def nameprep(self, name: str) -> str:
        """Convert Python name to HTML attribute/tag name."""
        # Undo keyword mangling (class_ -> class, for_ -> for)
        name = Element.PYTHON_KWORD_MAP.get(name, name)

        # Add namespace prefix if applicable
        if self.builder.namespace:
            name = self.builder.namespace[0] + ':' + name

        # Convert double underscore to colon (for namespaces)
        # Convert single underscore to hyphen (for web components, data- attrs, etc.)
        return name.replace('__', ':').replace('_', '-')


########################################################################
# Example/Demo code
########################################################################
if __name__ == "__main__":
    # Demo: Create a simple HTML page
    myname = 'HASSAN'

    with Builder() as xml:
        xml.write_doctype("html")
        with xml.html(lang="en"):
            with xml.head:
                xml.meta(SELFCLOSING, charset="UTF-8")
                xml.meta(SELFCLOSING, name="viewport", content="width=device-width, initial-scale=1.0")
                xml.title("py2html Demo")
                xml.style("""
                    body { font-family: Arial, sans-serif; margin: 2rem; }
                    .container { max-width: 800px; margin: 0 auto; }
                    h1 { color: #333; }
                    .highlight { background-color: #ffffcc; padding: 0.2rem 0.5rem; }
                """)
            with xml.body:
                with xml.div(class_="container"):
                    xml.h1(f"Hello, {myname}!")
                    xml.p("This is a demo of the py2html library.")

                    with xml.p:
                        xml.write("You can mix text with ")
                        xml.span("highlighted content", class_="highlight")
                        xml.write(" easily.")

                    xml.h2("Features")
                    with xml.ul:
                        xml.li("Context manager syntax for nesting")
                        xml.li("Automatic HTML escaping")
                        xml.li("Self-closing tag support")
                        xml.li("Namespace support for XML/SVG")

                    xml.hr(SELFCLOSING)

                    with xml.p:
                        xml.write("Special characters are escaped: ")
                        xml.code("< > & \" '")

                    # Self-closing tags
                    xml.br(SELFCLOSING)
                    xml.img(SELFCLOSING, src="https://via.placeholder.com/150", alt="Placeholder image")

    html_output = str(xml)
    print(html_output)

    # Preview - using browser which always works
    print("\n" + "=" * 60)
    print("Opening preview in browser...")
    print("=" * 60)
    preview_in_browser(html_output)
