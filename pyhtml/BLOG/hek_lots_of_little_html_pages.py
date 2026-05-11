#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lots of Little HTML Pages — inspired by Jim Nielsen's blog post:
https://blog.jim-nielsen.com/2025/lots-of-little-html-pages/

Instead of building JavaScript-powered interactions, build separate HTML pages
linked together with CSS cross-document view transitions.

Generates a set of interconnected HTML pages:
  - index.html      (homepage with post list, "recent" filter active)
  - popular.html     (posts filtered by popularity)
  - hn.html          (posts filtered by HN traffic)
  - menu.html        (full-page navigation menu)
  - search.html      (full-page search)

Run: python hek_lots_of_little_html_pages.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from py2html import Builder, SELFCLOSING
from contextlib import contextmanager
import webbrowser
import tempfile

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "lots_of_little_html_pages")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared data ──────────────────────────────────────────────────────

POSTS_RECENT = [
    ("View Transitions Are a Game Changer", "2025-03-10", 42),
    ("CSS Anchor Positioning Deep Dive", "2025-03-08", 38),
    ("The Return of Server-Side Rendering", "2025-03-05", 91),
    ("Building Without a Bundler", "2025-02-28", 27),
    ("HTML Dialog Element in Practice", "2025-02-20", 63),
    ("Web Components: One Year Later", "2025-02-15", 55),
]

POSTS_POPULAR = sorted(POSTS_RECENT, key=lambda p: p[2], reverse=True)

POSTS_HN = [p for p in POSTS_RECENT if p[2] > 50]

NAV_ITEMS = [
    ("Home", "index.html"),
    ("Archive", "#"),
    ("About", "#"),
    ("Notes", "#"),
    ("Projects", "#"),
    ("Feeds", "#"),
]


# ── Shared styles ────────────────────────────────────────────────────

SHARED_CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; color: #1a1a1a; background: #fafafa;
    max-width: 640px; margin: 0 auto; padding: 1.5rem;
}
a { color: #0055cc; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Cross-document view transitions */
@view-transition { navigation: auto; }

::view-transition-old(root) {
    animation: fade-out 0.15s ease-out;
}
::view-transition-new(root) {
    animation: fade-in 0.2s ease-in;
}
@keyframes fade-out { to { opacity: 0; } }
@keyframes fade-in  { from { opacity: 0; } }
"""

HEADER_CSS = """\
header {
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 1rem; border-bottom: 1px solid #e0e0e0;
    margin-bottom: 1.5rem;
}
header .site-title { font-size: 1.1rem; font-weight: 700; }
header nav { display: flex; gap: 0.75rem; }
header nav a {
    font-size: 0.85rem; color: #555;
    padding: 0.25rem 0.5rem; border-radius: 4px;
}
header nav a:hover { background: #eee; text-decoration: none; }
"""

TABS_CSS = """\
.tabs {
    display: flex; gap: 0.25rem; margin-bottom: 1.5rem;
    border-bottom: 2px solid #e0e0e0; padding-bottom: 0;
}
.tabs a {
    padding: 0.5rem 1rem; font-size: 0.9rem; color: #555;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: color 0.15s;
}
.tabs a:hover { color: #1a1a1a; text-decoration: none; }
.tabs a.active {
    color: #0055cc; border-bottom-color: #0055cc; font-weight: 600;
}

/* View transition for the post list */
.post-list { view-transition-name: post-list; }

::view-transition-old(post-list) {
    animation: slide-out 0.2s ease-out;
}
::view-transition-new(post-list) {
    animation: slide-in 0.25s ease-in;
}
@keyframes slide-out { to   { opacity: 0; transform: translateY(8px); } }
@keyframes slide-in  { from { opacity: 0; transform: translateY(-8px); } }
"""

POST_LIST_CSS = """\
.post-list { list-style: none; }
.post-list li {
    padding: 0.75rem 0;
    border-bottom: 1px solid #f0f0f0;
    display: flex; justify-content: space-between; align-items: baseline;
}
.post-list li:last-child { border-bottom: none; }
.post-list .post-title { font-size: 0.95rem; }
.post-list .post-meta {
    font-size: 0.8rem; color: #888; white-space: nowrap; margin-left: 1rem;
}
"""

MENU_CSS = """\
.menu-overlay {
    view-transition-name: menu-overlay;
    min-height: 100vh; display: flex; flex-direction: column;
    justify-content: center; align-items: center;
}
.menu-overlay .close-btn {
    position: fixed; top: 1.5rem; right: 1.5rem;
    font-size: 1.5rem; color: #555; width: 2.5rem; height: 2.5rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; background: #f0f0f0;
}
.menu-overlay .close-btn:hover { background: #e0e0e0; text-decoration: none; }
.menu-overlay ul { list-style: none; text-align: center; }
.menu-overlay li { margin: 1rem 0; }
.menu-overlay li a { font-size: 1.5rem; font-weight: 600; color: #1a1a1a; }

::view-transition-old(menu-overlay) {
    animation: menu-out 0.2s ease-out;
}
::view-transition-new(menu-overlay) {
    animation: menu-in 0.25s ease-in;
}
@keyframes menu-out { to   { opacity: 0; transform: scale(0.97); } }
@keyframes menu-in  { from { opacity: 0; transform: scale(1.03); } }
"""

SEARCH_CSS = """\
.search-page {
    view-transition-name: search-page;
    padding-top: 3rem;
}
.search-page .close-btn {
    position: fixed; top: 1.5rem; right: 1.5rem;
    font-size: 1.5rem; color: #555; width: 2.5rem; height: 2.5rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; background: #f0f0f0;
}
.search-page .close-btn:hover { background: #e0e0e0; text-decoration: none; }
.search-page h2 { margin-bottom: 1rem; font-size: 1.3rem; }
.search-page input[type="search"] {
    width: 100%; padding: 0.75rem 1rem; font-size: 1.1rem;
    border: 2px solid #ddd; border-radius: 8px;
    outline: none; transition: border-color 0.15s;
}
.search-page input[type="search"]:focus { border-color: #0055cc; }
.search-page .search-hint {
    margin-top: 0.75rem; font-size: 0.85rem; color: #888;
}

::view-transition-old(search-page) {
    animation: search-out 0.2s ease-out;
}
::view-transition-new(search-page) {
    animation: search-in 0.25s ease-in;
}
@keyframes search-out { to   { opacity: 0; transform: translateY(-12px); } }
@keyframes search-in  { from { opacity: 0; transform: translateY(12px); } }
"""


# ── Page skeleton decorator ─────────────────────────────────────────

@contextmanager
def page(title, *css_parts):
    """Yield (xml, body) with doctype/html/head/style/body already open."""
    with Builder() as xml:
        xml.write_doctype("html")
        with xml.html(lang="en"):
            with xml.head:
                xml.meta(SELFCLOSING, charset="UTF-8")
                xml.meta(SELFCLOSING, name="viewport", content="width=device-width, initial-scale=1.0")
                xml.title(title)
                xml.style(SHARED_CSS + "".join(css_parts))
            with xml.body:
                yield xml
    # after yield, caller can do str(xml)
    page._last = xml

def write_page(filename, title, *css_parts):
    """Decorator: the decorated function receives xml inside an open <body>.
    The finished HTML is written to OUTPUT_DIR/filename."""
    def decorator(fn):
        with page(title, *css_parts) as xml:
            fn(xml)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(page._last))
        return path
    return decorator


# ── Shared components ────────────────────────────────────────────────

def build_header(xml):
    with xml.header:
        xml.a("jim-nielsen.com", href="index.html", class_="site-title")
        with xml.nav:
            xml.a("Search", href="search.html")
            xml.a("Menu", href="menu.html")


# ── Page builders ────────────────────────────────────────────────────

def build_post_list_page(filename, active_tab, posts):
    @write_page(filename, f"{active_tab} Posts — Jim Nielsen's Blog",
                HEADER_CSS, TABS_CSS, POST_LIST_CSS)
    def _(xml):
        build_header(xml)
        xml.h1("Posts")

        tabs = [("Recent", "index.html"), ("Popular", "popular.html"), ("HN", "hn.html")]
        with xml.div(class_="tabs"):
            for label, href in tabs:
                if label == active_tab:
                    xml.a(label, href=href, class_="active")
                else:
                    xml.a(label, href=href)

        with xml.ul(class_="post-list"):
            for title, date, points in posts:
                with xml.li:
                    xml.a(title, href="#", class_="post-title")
                    if active_tab == "HN":
                        xml.span(f"{points} pts", class_="post-meta")
                    else:
                        xml.span(date, class_="post-meta")
    return _


@write_page("menu.html", "Menu — Jim Nielsen's Blog", MENU_CSS)
def menu_page(xml):
    with xml.div(class_="menu-overlay"):
        xml.a("×", href="index.html", class_="close-btn")
        with xml.ul:
            for label, href in NAV_ITEMS:
                with xml.li:
                    xml.a(label, href=href)


@write_page("search.html", "Search — Jim Nielsen's Blog", SEARCH_CSS)
def search_page(xml):
    with xml.div(class_="search-page"):
        xml.a("×", href="index.html", class_="close-btn")
        xml.h2("Search")
        xml.input(SELFCLOSING, type="search", placeholder="Search posts...", autofocus=True)
        xml.p("Try searching for CSS, HTML, or transitions.", class_="search-hint")


# ── Generate all pages ───────────────────────────────────────────────

if __name__ == "__main__":
    pages = [
        build_post_list_page("index.html",   "Recent",  POSTS_RECENT),
        build_post_list_page("popular.html",  "Popular", POSTS_POPULAR),
        build_post_list_page("hn.html",       "HN",      POSTS_HN),
        menu_page,
        search_page,
    ]

    print(f"Generated {len(pages)} pages in {OUTPUT_DIR}/")
    for p in pages:
        print(f"  {os.path.basename(p)}")

    index = os.path.join(OUTPUT_DIR, "index.html")
    webbrowser.open(f"file://{index}")
    print(f"\nOpened {index} in browser.")
    print("Click tabs (Recent/Popular/HN), Menu, and Search to see cross-document view transitions.")
