#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A rewrite of Jim Nielsen's "Building Websites With LLMS" blog post
(https://blog.jim-nielsen.com/2025/lots-of-little-html-pages/)
rendered as an HTML page, using the py2html Builder and referencing
the companion code (hek_lots_of_little_html_pages.py) as live examples.

Run:  python hek_lots_of_little_html_pages_blog.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from py2html import Builder, SELFCLOSING, preview_in_browser
import textwrap

# ── Syntax highlighting (minimal, CSS-only via <span> classes) ───────

def code_block(xml, code, lang="python"):
    with xml.pre:
        xml.code(textwrap.dedent(code).strip(), class_=f"language-{lang}")


def inline_code(xml, text):
    xml.code(text, class_="inline")


# ── Build the blog post ─────────────────────────────────────────────

with Builder() as xml:
    xml.write_doctype("html")
    with xml.html(lang="en"):
        with xml.head:
            xml.meta(SELFCLOSING, charset="UTF-8")
            xml.meta(SELFCLOSING, name="viewport", content="width=device-width, initial-scale=1.0")
            xml.title("Building Websites With LLMS (Lots of Little HTML Pages)")
            xml.style("""\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.8; color: #1a1a1a; background: #fdfdfd;
    max-width: 680px; margin: 0 auto; padding: 2rem 1.5rem 4rem;
}
h1 { font-size: 2rem; line-height: 1.25; margin-bottom: 0.25rem; }
h2 { font-size: 1.4rem; margin-top: 2.5rem; margin-bottom: 0.75rem; border-bottom: 1px solid #e8e8e8; padding-bottom: 0.4rem; }
h3 { font-size: 1.1rem; margin-top: 1.75rem; margin-bottom: 0.5rem; }
p  { margin-bottom: 1rem; }
a  { color: #0055cc; }
ul, ol { margin: 0 0 1rem 1.5rem; }
li { margin-bottom: 0.35rem; }
blockquote {
    border-left: 3px solid #ccc; margin: 1.25rem 0; padding: 0.5rem 1rem;
    color: #555; font-style: italic;
}
pre {
    background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px;
    padding: 1rem; overflow-x: auto; margin: 1rem 0 1.5rem;
    font-size: 0.85rem; line-height: 1.5;
}
code { font-family: "SF Mono", Menlo, Consolas, monospace; }
code.inline {
    background: #f0f0f0; padding: 0.15rem 0.4rem; border-radius: 3px;
    font-size: 0.9em;
}
.meta { color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
.footnote { font-size: 0.85rem; color: #666; margin-top: 3rem; border-top: 1px solid #e0e0e0; padding-top: 1rem; }
img.gif-demo {
    display: block; max-width: 100%; border: 1px solid #e0e0e0;
    border-radius: 6px; margin: 1rem 0;
}
.key-idea {
    background: #f0f7ff; border-left: 4px solid #0055cc;
    padding: 0.75rem 1rem; margin: 1.25rem 0; border-radius: 0 6px 6px 0;
}
.diagram {
    background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
    padding: 1rem; margin: 1rem 0; font-family: "SF Mono", Menlo, monospace;
    font-size: 0.82rem; line-height: 1.6; white-space: pre; overflow-x: auto;
}
""")

        with xml.body:

            # ── Title & meta ─────────────────────────────────────
            xml.h1("Building Websites With LLMS")
            with xml.p(class_="meta"):
                xml.write("March 4, 2025 ")
                xml.write(" | Based on ")
                xml.a("Jim Nielsen's original post",
                       href="https://blog.jim-nielsen.com/2025/lots-of-little-html-pages/")
                xml.write(", rewritten with working py2html code examples")

            # ── Intro ────────────────────────────────────────────
            with xml.p:
                xml.write("And by LLMS I mean: ")
                xml.strong("(L)ots of (L)ittle ht(M)l page(S).")

            with xml.p:
                xml.write("With cross-document view transitions getting broader support, ")
                xml.write("building in-page, progressively-enhanced interactions with JavaScript is ")
                xml.em("more work")
                xml.write(" than simply building two HTML pages and linking them.")

            with xml.p:
                xml.write('I\'m calling this approach "lots of little HTML pages." ')
                xml.write("Whenever I find myself reaching for JavaScript to build a fly-out menu, ")
                xml.write("on-page search, or content filter, I stop and ask: ")

            with xml.blockquote:
                xml.p("Can I build this as a separate HTML page triggered by a link, rather than JavaScript-injected content built from a button?")

            with xml.p:
                xml.write("The results are great. Separate, small HTML pages for each interaction, ")
                xml.write("CSS transitions handling the polish, less code to maintain, and it feels better than JS.")

            xml.p("Let me show you with real, runnable code.")

            # ── The approach: a diagram ──────────────────────────
            xml.h2("The Architecture")

            with xml.p:
                xml.write("Instead of one page with JS toggling visibility, you get a folder of linked HTML files:")

            xml.div("""\
lots_of_little_html_pages/
  index.html        <-- post list (Recent tab active)
  popular.html      <-- same layout, sorted by popularity
  hn.html           <-- filtered to high-HN-traffic posts
  menu.html         <-- full-page navigation overlay
  search.html       <-- full-page search""", class_="diagram")

            with xml.p:
                xml.write("Every page is a complete HTML document. Navigation between them is just ")
                inline_code(xml, "<a href>")
                xml.write(" links. The magic is a single CSS rule:")

            code_block(xml, """
                @view-transition { navigation: auto; }
            """, lang="css")

            with xml.p:
                xml.write("That one line tells the browser to animate between pages automatically. ")
                xml.write("Add named ")
                inline_code(xml, "view-transition-name")
                xml.write(" properties to specific elements and you get targeted, smooth transitions.")

            # ── Example 1: Filtering ─────────────────────────────
            xml.h2("Example 1: Filtering")

            with xml.p:
                xml.write("I wanted a list of posts filtered by different criteria: most recent, ")
                xml.write("most popular, and top Hacker News posts. My first impulse was client-side ")
                xml.write("JavaScript: stick a bunch of ")
                inline_code(xml, "<li>")
                xml.write("s in the DOM, show/hide based on the current filter.")

            with xml.p:
                xml.write("But it got complicated fast. Each list needed different data and sort orders. ")
                xml.write("What started as a quick toggle turned into ")
                inline_code(xml, "data-x")
                xml.write(" attributes, per-list sorting logic, and a lot of client-side code for what ")
                xml.write("my static site generator could do in seconds.")

            with xml.p:
                xml.write("Then I thought: why not just make each filter its own HTML page?")

            xml.h3("The data")

            with xml.p:
                xml.write("Each page draws from the same post list, just sorted/filtered differently:")

            code_block(xml, """
                POSTS_RECENT = [
                    ("View Transitions Are a Game Changer", "2025-03-10", 42),
                    ("CSS Anchor Positioning Deep Dive",    "2025-03-08", 38),
                    ("The Return of Server-Side Rendering", "2025-03-05", 91),
                    ("Building Without a Bundler",          "2025-02-28", 27),
                    ("HTML Dialog Element in Practice",     "2025-02-20", 63),
                    ("Web Components: One Year Later",      "2025-02-15", 55),
                ]

                POSTS_POPULAR = sorted(POSTS_RECENT, key=lambda p: p[2], reverse=True)
                POSTS_HN = [p for p in POSTS_RECENT if p[2] > 50]
            """)

            xml.h3("Eliminating boilerplate with a decorator")

            with xml.p:
                xml.write("Every page needs the same skeleton: doctype, ")
                inline_code(xml, "<html>")
                xml.write(", ")
                inline_code(xml, "<head>")
                xml.write(" with meta tags and styles, then ")
                inline_code(xml, "<body>")
                xml.write(". That's 8 lines of boilerplate per page. A ")
                inline_code(xml, "@write_page")
                xml.write(" decorator eliminates all of it:")

            code_block(xml, """
                @contextmanager
                def page(title, *css_parts):
                    \"\"\"Yield xml with doctype/html/head/style/body already open.\"\"\"
                    with Builder() as xml:
                        xml.write_doctype("html")
                        with xml.html(lang="en"):
                            with xml.head:
                                xml.meta(SELFCLOSING, charset="UTF-8")
                                xml.meta(SELFCLOSING, name="viewport",
                                         content="width=device-width, initial-scale=1.0")
                                xml.title(title)
                                xml.style(SHARED_CSS + "".join(css_parts))
                            with xml.body:
                                yield xml
                    page._last = xml

                def write_page(filename, title, *css_parts):
                    \"\"\"Decorator: function receives xml inside an open <body>.
                    Finished HTML is written to OUTPUT_DIR/filename.\"\"\"
                    def decorator(fn):
                        with page(title, *css_parts) as xml:
                            fn(xml)
                        path = os.path.join(OUTPUT_DIR, filename)
                        with open(path, "w") as f:
                            f.write(str(page._last))
                        return path
                    return decorator
            """)

            with xml.p:
                xml.write("Now the decorated function only deals with what's inside ")
                inline_code(xml, "<body>")
                xml.write(". Here's the tab page builder:")

            xml.h3("Building a tab page")

            code_block(xml, """
                def build_post_list_page(filename, active_tab, posts):
                    @write_page(filename, f"{active_tab} Posts — Jim Nielsen's Blog",
                                HEADER_CSS, TABS_CSS, POST_LIST_CSS)
                    def _(xml):
                        build_header(xml)
                        xml.h1("Posts")

                        tabs = [("Recent", "index.html"),
                                ("Popular", "popular.html"),
                                ("HN", "hn.html")]
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
                                    xml.span(date, class_="post-meta")
                    return _
            """)

            with xml.p:
                xml.write("Then generate all three pages with one call each:")

            code_block(xml, """
                build_post_list_page("index.html",   "Recent",  POSTS_RECENT)
                build_post_list_page("popular.html",  "Popular", POSTS_POPULAR)
                build_post_list_page("hn.html",       "HN",      POSTS_HN)
            """)

            xml.h3("The transition CSS")

            with xml.p:
                xml.write("The post list element gets a ")
                inline_code(xml, "view-transition-name")
                xml.write(" so the browser knows to animate it specifically when navigating between tabs:")

            code_block(xml, """
                /* Every page opts in to cross-document transitions */
                @view-transition { navigation: auto; }

                /* The post list slides in/out when switching tabs */
                .post-list { view-transition-name: post-list; }

                ::view-transition-old(post-list) {
                    animation: slide-out 0.2s ease-out;
                }
                ::view-transition-new(post-list) {
                    animation: slide-in 0.25s ease-in;
                }
                @keyframes slide-out { to   { opacity: 0; transform: translateY(8px); } }
                @keyframes slide-in  { from { opacity: 0; transform: translateY(-8px); } }
            """, lang="css")

            with xml.div(class_="key-idea"):
                with xml.p:
                    xml.write("The tabs aren't toggling visibility. Each click is a full page navigation ")
                    xml.write("to a separate HTML document. The browser's view transition API makes it ")
                    xml.em("look")
                    xml.write(" like an in-page interaction.")

            # ── Example 2: Navigation ────────────────────────────
            xml.h2("Example 2: Navigation")

            with xml.p:
                xml.write("Usually I'd think: hamburger icon, hidden nav drawer, JS to reveal it, ")
                xml.write("ARIA attributes, focus trapping... But what if the menu is just a new HTML page?")

            with xml.p:
                xml.write("With the ")
                inline_code(xml, "@write_page")
                xml.write(" decorator, this is just the body content:")

            code_block(xml, """
                @write_page("menu.html", "Menu — Jim Nielsen's Blog", MENU_CSS)
                def menu_page(xml):
                    with xml.div(class_="menu-overlay"):
                        xml.a("\\u00d7", href="index.html", class_="close-btn")
                        with xml.ul:
                            for label, href in NAV_ITEMS:
                                with xml.li:
                                    xml.a(label, href=href)
            """)

            with xml.p:
                xml.write("The menu overlay gets its own view transition name. When you click ")
                xml.write('"Menu" in the header, the browser navigates to ')
                inline_code(xml, "menu.html")
                xml.write(" and the CSS transition makes it feel like a slide-in overlay:")

            code_block(xml, """
                .menu-overlay { view-transition-name: menu-overlay; }

                ::view-transition-old(menu-overlay) {
                    animation: menu-out 0.2s ease-out;
                }
                ::view-transition-new(menu-overlay) {
                    animation: menu-in 0.25s ease-in;
                }
                @keyframes menu-out { to   { opacity: 0; transform: scale(0.97); } }
                @keyframes menu-in  { from { opacity: 0; transform: scale(1.03); } }
            """, lang="css")

            with xml.p:
                xml.write('Clicking the "x" navigates back to ')
                inline_code(xml, "index.html")
                xml.write(". No JavaScript, no state management, no event listeners.")

            # ── Example 3: Search ────────────────────────────────
            xml.h2("Example 3: Search")

            with xml.p:
                xml.write("Same idea. The search icon in the header doesn't open a JS-powered modal. ")
                xml.write("It's a link to ")
                inline_code(xml, "search.html")
                xml.write(":")

            code_block(xml, """
                @write_page("search.html", "Search — Jim Nielsen's Blog", SEARCH_CSS)
                def search_page(xml):
                    with xml.div(class_="search-page"):
                        xml.a("\\u00d7", href="index.html", class_="close-btn")
                        xml.h2("Search")
                        xml.input(SELFCLOSING, type="search",
                                  placeholder="Search posts...",
                                  autofocus=True)
            """)

            with xml.p:
                xml.write("The ")
                inline_code(xml, "autofocus=True")
                xml.write(" attribute means the search input is focused as soon as the page loads. ")
                xml.write("No JS needed for that either.")

            # ── Putting it all together ──────────────────────────
            xml.h2("Putting It All Together")

            with xml.p:
                xml.write("The entire generator is ~100 lines of Python. Five pages, zero JavaScript:")

            with xml.p:
                xml.write("The ")
                inline_code(xml, "menu_page")
                xml.write(" and ")
                inline_code(xml, "search_page")
                xml.write(" are generated at import time by their decorators. ")
                xml.write("The tab pages are generated by calling ")
                inline_code(xml, "build_post_list_page")
                xml.write(":")

            code_block(xml, """
                if __name__ == "__main__":
                    pages = [
                        build_post_list_page("index.html",   "Recent",  POSTS_RECENT),
                        build_post_list_page("popular.html",  "Popular", POSTS_POPULAR),
                        build_post_list_page("hn.html",       "HN",      POSTS_HN),
                        menu_page,    # already generated by @write_page
                        search_page,  # already generated by @write_page
                    ]
                    webbrowser.open(f"file://{OUTPUT_DIR}/index.html")
            """)

            with xml.p:
                xml.write("Run it:")

            code_block(xml, """
                python hek_lots_of_little_html_pages.py
            """, lang="bash")

            with xml.p:
                xml.write("Click between tabs, open the menu, try search. Every interaction is a ")
                xml.write("full page navigation, but it feels instant thanks to view transitions.")

            # ── Why this works ───────────────────────────────────
            xml.h2("Why This Works")

            with xml.ul:
                with xml.li:
                    xml.strong("Less code.")
                    xml.write(" No toggle state, no event listeners, no ARIA dance. Each page is self-contained HTML.")
                with xml.li:
                    xml.strong("Easy to maintain.")
                    xml.write(" Changing the menu means editing one function that generates one HTML file.")
                with xml.li:
                    xml.strong("Works without JS.")
                    xml.write(" The transitions are progressive enhancement. Without them, you still get working links.")
                with xml.li:
                    xml.strong("Fast.")
                    xml.write(" Each page is a small, static HTML file. No framework, no bundle, no hydration.")
                with xml.li:
                    xml.strong("Accessible by default.")
                    xml.write(" Standard links, standard page navigations. Screen readers, keyboard users, and bots all understand this.")

            with xml.div(class_="key-idea"):
                with xml.p:
                    xml.write("The key insight: it's really easy to build a simple website when you shift your ")
                    xml.write("perspective to viewing on-page interactivity as HTML page navigations powered ")
                    xml.write("by cross-document CSS transitions, rather than doing all of that in client-side JS.")

            # ── Footnote ─────────────────────────────────────────
            with xml.div(class_="footnote"):
                with xml.p:
                    xml.write("This post is a rewrite of ")
                    xml.a("Jim Nielsen's original",
                           href="https://blog.jim-nielsen.com/2025/lots-of-little-html-pages/")
                    xml.write(". The code examples come from ")
                    inline_code(xml, "hek_lots_of_little_html_pages.py")
                    xml.write(", which generates a working demo using ")
                    xml.a("py2html", href="https://github.com/anthropics/py2html")
                    xml.write(". Run it to see the transitions in action.")


# ── Output ───────────────────────────────────────────────────────────

html_output = str(xml)
filepath = preview_in_browser(html_output, "lots_of_little_html_pages_blog.html")
print(f"Blog post opened in browser: {filepath}")
